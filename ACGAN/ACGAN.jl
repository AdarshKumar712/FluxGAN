using Base.Iterators: partition
using Flux
using Flux.Optimise: update!
using Flux: logitbinarycrossentropy, logitcrossentropy
using Images
using MLDatasets
using Statistics
using Parameters: @with_kw
using Random
using Printf
using CUDAapi
using Zygote
if has_cuda()		# Check if CUDA is available
    @info "CUDA is on"
    import CuArrays		# If CUDA is available, import CuArrays
    CuArrays.allowscalar(false)
end

@with_kw struct HyperParams
    batch_size::Int = 128
    latent_dim::Int = 100
    nclasses::Int = 10
    epochs::Int = 30
    verbose_freq::Int = 1500
    output_x::Int = 10
    output_y::Int = 10
    lr_dscr::Float64 = 0.0001
    lr_gen::Float64 = 0.0001
end

mutable struct discriminator
    discr
    aux
    model
end

function discriminator(args)
    model = Chain(Conv((3,3), 1=>128, pad=(1,1), stride = (2,2)),
                  x->leakyrelu.(x, 0.2f0),
                  Dropout(0.4),
                  Conv((3,3), 128=>128, pad= (1,1), stride = (2,2), leakyrelu),
                  x->leakyrelu.(x, 0.2f0),
                  x -> reshape(x, :, size(x, 4)),
                  Dropout(0.4)) |> gpu
    discr = Dense(6272, 1, sigmoid) |> gpu
    aux = Chain(Dense(6272, 10), softmax) |> gpu
    discriminator(discr,aux,model)
end

mutable struct generator
    m1
    m2
    model
end

function generator(args)
    m1 = Chain(Dense(args.nclasses, 49), x-> reshape(x, 7 , 7 , 1 , size(x,2))) |> gpu
    m2 = Chain(Dense(args.latent_dim, 6272), x->leakyrelu.(x, 0.2f0), x-> reshape(x, 7,7,128, size(x,2))) |> gpu
    model = Chain(ConvTranspose((4, 4), 129 => 128; stride = 2, pad = 1),
            BatchNorm(128, leakyrelu),
            Dropout(0.25),
            ConvTranspose((4, 4), 128 => 64; stride = 2, pad = 1),
            BatchNorm(64, leakyrelu),
            Conv((7, 7), 64 => 1, tanh; stride = 1, pad = 3)) |> gpu
    return generator(m1, m2, model)
end

function (m::generator)(x, y)
    t = cat(m.m1(x), m.m2(y), dims = 3)
    m.model(t)
end

function load_data(hparams)
    # Load MNIST dataset
    images, labels = MLDatasets.FashionMNIST.traindata(Float32)
    # Normalize to [-1, 1] and convert it to WHCN
    image_tensor = permutedims(reshape(@.(2f0 * images - 1f0), 28, 28, 1, :), (2, 1, 3, 4))
    y = float.(Flux.onehotbatch(labels, 0:hparams.nclasses-1))
    # Partition into batches
    data = [(image_tensor[:, :, :, r], y[:,r]) |> gpu for r in partition(1:60000, hparams.batch_size)]
    return data
end

# Loss functions
function discr_loss(real_output, fake_output)
    real_loss = mean(logitbinarycrossentropy.(real_output, 1f0))
    fake_loss = mean(logitbinarycrossentropy.(fake_output, 0f0))
    return real_loss + fake_loss
end

function aux_loss(output, labels)
   return logitcrossentropy(output, labels) 
end

generator_loss(fake_output) = mean(logitbinarycrossentropy.(fake_output, 1f0))

function train_discr(discr, fake_data, original_data ,labels, opt_discr)
    ps = params(discr.aux, discr.discr,discr.model)
    loss = 0.0
    for i in 1:2
    loss, back = Zygote.pullback(ps) do
           original_common = discr.model(original_data) |> gpu
           fake_common =  discr.model(fake_data) |> gpu
           d1 = discr_loss(discr.discr(original_common), discr.discr(fake_common))
           x = discr.aux(original_common)
           a1 = aux_loss(x, labels |> gpu)
           d1+ a1
           end
    grads = back(1f0)
    update!(opt_discr, ps, grads)
    end
    return loss
end

Zygote.@nograd train_discr

function train_gan(gen, discr, original_data, label, opt_gen, opt_discr, hparams)
    noise = randn!(similar(original_data, (hparams.latent_dim, hparams.batch_size))) |> gpu
    ps = params(gen.m1,gen.m2, gen.model)
    loss = Dict()
    loss["gen"], back = Zygote.pullback(ps) do
            fake = gen(label, noise) |> gpu
            loss["discr"] = train_discr(discr, fake, original_data, label, opt_discr)
            fake_common = discr.model(fake)
            generator_loss(discr.discr(fake_common)) + aux_loss(discr.aux(fake_common), label)
    end
    grads = back(1f0)
    update!(opt_gen, ps, grads)
    return loss
end


function create_output_image(gen, fixed_noise, fixed_labels, hparams)
    @eval Flux.istraining() = false
    fake_images = @. cpu(gen(fixed_labels, fixed_noise))
    @eval Flux.istraining() = true
    image_array = dropdims(reduce(vcat, reduce.(hcat, partition(fake_images, hparams.output_y))); dims=(3, 4))
    image_array = @. Gray(image_array + 1f0) / 2f0
    return image_array
end


function train()
    hparams = HyperParams()

    data = load_data(hparams)

    fixed_noise = [randn(hparams.latent_dim, 1) |> gpu for _=1:hparams.output_x*hparams.output_y]

    fixed_labels = [float.(Flux.onehotbatch(rand(0:hparams.nclasses-1,1), 0:hparams.nclasses-1)) |> gpu for _=1:hparams.output_x*hparams.output_y]

    # Discriminator
    dscr = discriminator(hparams) 

    # Generator
    gen =  generator(hparams)

    # Optimizers
    opt_dscr = ADAM(hparams.lr_dscr, (0.5,0.999))
    opt_gen = ADAM(hparams.lr_gen, (0.5,0.999))

    isdir("output")||mkdir("output")

    # Training
    train_steps = 0
    for ep in 1:hparams.epochs
        @info "Epoch $ep"
        for (x,y) in data[1:468]
            # Update discriminator and generator
            loss = train_gan(gen, dscr, x, y, opt_gen, opt_dscr, hparams)

            if train_steps % hparams.verbose_freq == 0
                @info("Train step $(train_steps), Discriminator loss = $(loss["discr"]), Generator loss = $(loss["gen"])")
                # Save generated fake image
                output_image = create_output_image(gen, fixed_noise, fixed_labels, hparams)
                save(@sprintf("output/cgan_steps_%06d.png", train_steps), output_image)
            end
            train_steps += 1
        end
    end

    output_image = create_output_image(gen, fixed_noise, fixed_labels, hparams)
    save(@sprintf("output/cgan_steps_%06d.png", train_steps), output_image)
    return fixed_labels
end    

cd(@__DIR__)
fixed_labels = train()
