# Ref:https://machinelearningmastery.com/least-squares-generative-adversarial-network/

using Base.Iterators: partition
using Flux
using Flux.Optimise: update!
using Flux: mse
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
    epochs::Int = 20
    verbose_freq::Int = 1000
    output_x::Int = 6        # No. of sample images to concatenate along x-axis 
    output_y::Int = 6        # No. of sample images to concatenate along y-axis
    lr_dscr::Float64 = 0.0002
    lr_gen::Float64 = 0.0002
end

function generator(args)
    n_nodes = 256*7*7
    return Chain(Dense(args.latent_dim, n_nodes),BatchNorm(n_nodes),x -> leakyrelu.(x,0.2f0), x-> reshape(x, 7,7,256, size(x,2)),
                 ConvTranspose((4,4), 256=>128, stride=(2,2), pad=(1,1)), BatchNorm(128), x -> leakyrelu.(x,0.2f0),
                 ConvTranspose((4,4), 128=>64, stride=(2,2), pad = (1,1)), BatchNorm(64), x -> leakyrelu.(x,0.2f0),
                 Conv((7,7), 64=>1, stride=1, pad=3, tanh)) |> gpu
end

function discriminator(args)
    return  Chain(Conv((4,4), 1=>64, stride=2, pad=1), BatchNorm(64), x -> leakyrelu.(x,0.2f0),
                  Conv((4,4), 64=>128, stride=2, pad=1), BatchNorm(64), x -> leakyrelu.(x,0.2f0),
                  x->reshape(x, :, size(x,4)), Dense(6272, 1, sigmoid))|> gpu
end

function load_data(hparams)
    # Load MNIST dataset
    images, labels = MLDatasets.MNIST.traindata(Float32)
    # Normalize to [-1, 1] and convert it to WHCN
    image_tensor = permutedims(reshape(@.(2f0 * images - 1f0), 28, 28, 1, :), (2, 1, 3, 4))
    # Partition into batches
    data = [image_tensor[:,:,:,r] |> gpu for r in partition(1:60000, hparams.batch_size)]
    return data
end

# Loss functions
function discr_loss(real_output, fake_output)
    real_loss = mean(mse.(real_output, 1f0))
    fake_loss = mean(mse.(fake_output, 0f0))
    return (real_loss + fake_loss)
end

generator_loss(fake_output) = mean(mse.(fake_output, 1f0))

function train_discr(discr, original_data, fake_data, opt_discr)
    ps = Flux.params(discr)
    loss, back = Zygote.pullback(ps) do
                      discr_loss(discr(original_data), discr(fake_data))
    end
    grads = back(1f0)
    update!(opt_discr, ps, grads)
    return loss
end

Zygote.@nograd train_discr

function train_gan(gen, discr, original_data, opt_gen, opt_discr, hparams)
    noise = randn!(similar(original_data, (hparams.latent_dim, hparams.batch_size))) |> gpu
    loss = Dict()
    ps = Flux.params(gen)
    loss["gen"], back = Zygote.pullback(ps) do
                          fake_ = gen(noise)
                          loss["discr"] = train_discr(discr, original_data, fake_, opt_discr)
                          generator_loss(discr(fake_))
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
    gen =  generator(hparams) |> gpu

    # Optimizers
    opt_dscr = ADAM(hparams.lr_dscr)
    opt_gen = ADAM(hparams.lr_gen)

    isdir("output")||mkdir("output")

    # Training
    train_steps = 0
    for ep in 1:hparams.epochs
        @info "Epoch $ep"
        for x in data
            # Update discriminator and generator
            loss = train_gan(gen, dscr, x, opt_gen, opt_dscr, hparams)

            if train_steps % hparams.verbose_freq == 0
                @info("Train step $(train_steps), Discriminator loss = $(loss["discr"]), Generator loss = $(loss["gen"])")
                # Save generated fake image
                output_image = create_output_image(gen, fixed_noise, hparams)
                save(@sprintf("output/gan_steps_%06d.png", train_steps), output_image)
            end
            train_steps += 1
        end
    end

    output_image = create_output_image(gen, fixed_noise, hparams)
    save(@sprintf("output/gan_steps_%06d.png", train_steps), output_image)
end    

cd(@__DIR__)
train()
