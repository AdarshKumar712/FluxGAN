#Semi-Supervised GAN
# Ref: https://machinelearningmastery.com/semi-supervised-generative-adversarial-network/
# The approach is based on the definition of the semi-supervised model in the 2016 paper by Tim Salimans, 
# et al.from OpenAI titled “Improved Techniques for Training GANs.”

using Base.Iterators: partition
using Flux
using Flux.Optimise: update!
using Flux: binarycrossentropy, logitcrossentropy
using Images
using MLDatasets
using Statistics, StatsBase
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
    sample_each::Int = 5000
    epochs::Int = 20
    verbose_freq::Int = 1000
    output_x::Int = 10
    output_y::Int = 10
    lr_dscr::Float64 = 0.0002
    lr_gen::Float64 = 0.0002
end

function generator(args)
    return Chain(Dense(args.latent_dim, 6272), x->leakyrelu.(x, 0.2f0), x-> reshape(x, 7,7,128, size(x,2)),
            ConvTranspose((4, 4), 128 => 128; stride = 2, pad = 1),
            BatchNorm(128, leakyrelu),
            Dropout(0.25),
            ConvTranspose((4, 4), 128 => 64; stride = 2, pad = 1),
            BatchNorm(64, leakyrelu),
            Conv((7, 7), 64 => 1, tanh; stride = 1, pad = 3)) |> gpu
end

function discriminator(args)
    return Chain(Conv((3,3), 1=>128, pad=(1,1), stride = (2,2)),
                  x->leakyrelu.(x, 0.2f0),
                  Dropout(0.4),
                  Conv((3,3), 128=>128, pad= (1,1), stride = (2,2), leakyrelu),
                  x->leakyrelu.(x, 0.2f0),
                  x -> reshape(x, :, size(x, 4)),
                  Dropout(0.4), Dense(6272, 10)) |> gpu
end

function load_data(hparams)
    # Load MNIST dataset
    images, labels = MLDatasets.MNIST.traindata(Float32)
    # Normalize to [-1, 1] and convert it to WHCN
    image_tensor = permutedims(reshape(@.(2f0 * images - 1f0), 28, 28, 1, :), (2, 1, 3, 4))
    img, y = [],[]
    for i in 0:hparams.nclasses-1
        x = findall(==(i), labels)
        img_for_class = image_tensor[:,:,:,x]
        labels_for_class = labels[x]
        idxs = sample(1:length(x), hparams.sample_each, replace = false)
        [push!(img, img_for_class[:,:,:,idx]) for idx in idxs]
        push!(y, (labels_for_class[idxs]...))
    end
    imgs = zeros(28,28,1,length(y))
    for i in 1:length(img)
        imgs[:,:,:,i] = img[i]
    end
    n = length(y)
    y = float.(Flux.onehotbatch(y, 0:hparams.nclasses-1))
    idxs = sample(1:n, n, replace=false)
    img, y = imgs[:,:,:,idxs], y[:,idxs]
    data = [(img[:,:,:,r],y[:,r]) |> gpu for r in partition(1:n, hparams.batch_size)]
    return data
end

# Loss functions
function discr_loss(real_output, fake_output)
    real_loss = mean(binarycrossentropy.(real_output, 1f0))
    fake_loss = mean(binarycrossentropy.(fake_output, 0f0))
    return real_loss + fake_loss
end

function aux_loss(output, labels)
   return Flux.crossentropy(output, labels)
end

generator_loss(fake_output) = mean(binarycrossentropy.(fake_output, 1f0))

function custom_activation(output)
    exp_sum = sum(exp.(output))
    return exp_sum / (exp_sum + 1.0f0)
end

function train_discr(discr, fake_data, original_data ,labels, opt_discr)
    ps = params(discr)
    loss = 0.0
    for i in 1:2
    loss, back = Zygote.pullback(ps) do
           original_common = discr(original_data) |> gpu
           fake_common =  discr(fake_data) |> gpu
           d1 = discr_loss(custom_activation(original_common), custom_activation(fake_common))
           x = softmax(original_common) 
           a1 = aux_loss(x, labels)
           (d1 + a1)/2
           end
    grads = back(1f0)
    update!(opt_discr, ps, grads)
    end
    return loss
end

Zygote.@nograd train_discr

function train_gan(gen, discr, original_data, label, opt_gen, opt_discr, hparams)
    noise = randn!(similar(original_data, (hparams.latent_dim, hparams.batch_size))) |> gpu
    ps = params(gen)
    loss = Dict()
    loss["gen"], back = Zygote.pullback(ps) do
            fake = gen(noise) |> gpu
            loss["discr"] = train_discr(discr, fake, original_data, label, opt_discr)
            fake_common = discr(fake)
            generator_loss(custom_activation(fake_common))
    end
    grads = back(1f0)
    update!(opt_gen, ps, grads)
    return loss
end


function create_output_image(gen, fixed_noise, hparams)
    @eval Flux.istraining() = false
    fake_images = @. cpu(gen(fixed_noise))
    @eval Flux.istraining() = true
    image_array = dropdims(reduce(vcat, reduce.(hcat, partition(fake_images, hparams.output_y))); dims=(3, 4))
    image_array = @. Gray(image_array + 1f0) / 2f0
    return image_array
end


function train()
    hparams = HyperParams()

    data = load_data(hparams)

    fixed_noise = [randn(hparams.latent_dim, 1) |> gpu for _=1:hparams.output_x*hparams.output_y]

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
        for (x,y) in data
            # Update discriminator and generator
            loss = train_gan(gen, dscr, x, y, opt_gen, opt_dscr, hparams)

            if train_steps % hparams.verbose_freq == 0
                @info("Train step $(train_steps), Discriminator loss = $(loss["discr"]), Generator loss = $(loss["gen"])")
                # Save generated fake image
                output_image = create_output_image(gen, fixed_noise, hparams)
                save(@sprintf("output/sgan_steps_%06d.png", train_steps), output_image)
            end
            train_steps += 1
        end
    end

    output_image = create_output_image(gen, fixed_noise, fixed_labels, hparams)
    save(@sprintf("output/sgan_steps_%06d.png", train_steps), output_image)
end    

cd(@__DIR__)
train()
