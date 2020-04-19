using Base.Iterators: partition
using Flux
using Flux.Optimise: update!
using Flux: logitbinarycrossentropy, logitcrossentropy, mse
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
    code_dim::Int = 10
    epochs::Int = 20
    verbose_freq::Int = 1000
    output_x::Int = 6
    output_y::Int = 6
    lr_dscr::Float64 = 0.0001
    lr_gen::Float64 = 0.0001
end

struct generator
    model::Chain
end

function generator(args)
    input_dim = args.latent_dim + args.code_dim
    model = Chain(Dense(input_dim, 7*7*512), x->leakyrelu.(x, 0.2f0), x-> reshape(x, 7,7,512, size(x,2)),
        Conv((3,3), 512=>128, pad=1), BatchNorm(128), x->leakyrelu.(x, 0.2f0), 
        ConvTranspose((4,4), 128=>64, stride = (2,2), pad= 1), BatchNorm(64), x->leakyrelu.(x, 0.2f0),
        ConvTranspose((4,4), 64=>1, tanh,stride = 2, pad=1)) |> gpu
    return generator(model)
end

function (m::generator)(latent, code)
    t = cat(latent, code, dims = 1)
    m.model(t)
end

struct discriminator
    m_common
    d_model
    q_model
end

function discriminator(args)
    m_common = Chain(Conv((3,3), 1=>64, stride=2 , pad = 1),x->leakyrelu.(x, 0.2f0), Dropout(0.25),
                     Conv((3,3), 64=>128, stride = 2, pad=1), BatchNorm(128),x->leakyrelu.(x, 0.2f0),
                    Dropout(0.25), Conv((4,4), 128=>256, stride=2, pad=1), BatchNorm(256),x->leakyrelu.(x, 0.2f0),
                    Dropout(0.4), x->reshape(x, :,size(x,4))) |> gpu
    d_model = Chain(Dense(2304, 256),x->leakyrelu.(x, 0.2f0), Dense(256, 1, sigmoid)) |> gpu
    q_model = Chain(Dense(2304, 256),x->leakyrelu.(x, 0.2f0), Dense(256, args.code_dim), softmax) |> gpu
    return discriminator(m_common, d_model, q_model)
end

function load_data(hparams)
    # Load MNIST dataset
    images, labels = MLDatasets.MNIST.traindata(Float32)
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
    return (real_loss + fake_loss)/2
end

generator_loss(fake_output) = mean(logitbinarycrossentropy.(fake_output, 1f0))

function aux_loss(output, labels)
    return logitcrossentropy(output, labels)
end

function continuous_loss(output, true_val)
    return mse(output, true_val)
end

function train_discr(discr, fake_data, latent_code, original_data, opt_discr)
    ps = Flux.params(discr.m_common, discr.d_model, discr.q_model)
    loss, back = Zygote.pullback(ps) do
                       discr_loss(discr.d_model(discr.m_common(original_data)),discr.d_model(discr.m_common(fake_data))) + aux_loss(discr.q_model(discr.m_common(fake_data)), latent_code)*0.1f0
    end
    grads = back(1f0)
    update!(opt_discr, ps, grads)
    return loss
end

Zygote.@nograd train_discr

function train_gan(gen, discr, original_data, opt_gen, opt_discr, hparams)
    noise = randn!(similar(original_data, (hparams.latent_dim, hparams.batch_size))) |> gpu
    latent_code = float.(Flux.onehotbatch(rand(0:hparams.code_dim-1,hparams.batch_size), 0:hparams.code_dim-1)) |> gpu
    loss = Dict()
    ps = Flux.params(gen.model)
    loss["gen"], back = Zygote.pullback(ps) do
                         fake_ = gen(noise, latent_code)
                         loss["discr"] = train_discr(discr, fake_, latent_code, original_data, opt_discr)
                         t = discr.m_common(fake_)
                         generator_loss(discr.d_model(t)) + aux_loss(discr.q_model(t), latent_code)*0.1f0 
                 end
    grads = back(1f0)
    update!(opt_gen, ps, grads)
    return loss
end
    


function create_output_image(gen, fixed_noise, fixed_code, hparams)
    @eval Flux.istraining() = false
    fake_images = @. cpu(gen(fixed_noise, fixed_code))
    @eval Flux.istraining() = true
    image_array = dropdims(reduce(vcat, reduce.(hcat, partition(fake_images, hparams.output_y))); dims=(3, 4))
    image_array = @. Gray(image_array + 1f0) / 2f0
    return image_array
end

function train()
    hparams = HyperParams()

    data = load_data(hparams)

    fixed_noise = [randn(hparams.latent_dim, 1) |> gpu for _=1:hparams.output_x*hparams.output_y]

    # fixed_labels = [float.(Flux.onehotbatch(rand(0:hparams.nclasses-1,1), 0:hparams.nclasses-1)) |> gpu for _=1:hparams.output_x*hparams.output_y]

    fixed_code = [float.(Flux.onehotbatch(rand(0:hparams.code_dim-1,1), 0:hparams.code_dim-1))  |> gpu for _=1:hparams.output_x*hparams.output_y]
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
            loss = train_gan(gen, dscr, x, opt_gen, opt_dscr, hparams)
            
            if (isnan(loss["discr"]))
                 print(train_steps)
                 break
            end
            
            if train_steps % hparams.verbose_freq == 0
                @info("Train step $(train_steps), Discriminator loss = $(loss["discr"]), Generator loss = $(loss["gen"])")
                # Save generated fake image
                output_image = create_output_image(gen, fixed_noise, fixed_code, hparams)
                save(@sprintf("output/gan_steps_%06d.png", train_steps), output_image)
            end
            train_steps += 1
        end
    end

    output_image = create_output_image(gen, fixed_noise, fixed_code, hparams)
    save(@sprintf("output/infogan_steps_%06d.png", train_steps), output_image)
end    

cd(@__DIR__)
train()
