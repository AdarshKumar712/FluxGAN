using Base.Iterators: partition
using Flux
using Flux.Optimise: update!
using Flux: logitbinarycrossentropy, binarycrossentropy, mae
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
    latent_dim::Int = 10
    epochs::Int = 50
    verbose_freq::Int = 1000
    output_x::Int = 6        # No. of sample images to concatenate along x-axis 
    output_y::Int = 6        # No. of sample images to concatenate along y-axis
    lr_dscr::Float64 = 0.0002
    lr_gen::Float64 = 0.0002
end

function reparameterization(mu, logvar, args)
    std = exp.(logvar / 2) |> gpu
    sampled_z = Float32.(randn(size(mu)...)) |> gpu
    return sampled_z .* std .+ mu
end


struct encoder
    m
    mu
    logvar
end

function encoder(args)
    m =  Chain(x->reshape(x, :, size(x, 4)), Dense(784, 1024), x-> relu.(x),Dense(1024,1024), BatchNorm(1024), x->relu.(x)) |> gpu    
    mu = Dense(1024, args.latent_dim) |> gpu
    logvar = Dense(1024, args.latent_dim) |>gpu
    return encoder(m, mu, logvar)
end



function decoder(args)
    return Chain(Dense(args.latent_dim, 1024), x-> relu.(x), Dense(1024,1024),BatchNorm(1024), x-> relu.(x),
                 Dense(1024, 784, tanh), x-> reshape(x, 28,28,1,size(x,2))) |> gpu
end

function discriminator(args)
    return Chain(Dense(args.latent_dim, 1024), x-> leakyrelu.(x, 0.2f0), Dense(1024, 256),x-> leakyrelu.(x, 0.2f0),Dense(256, 1, sigmoid)) |> gpu
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
    real_loss = mean(binarycrossentropy.(real_output, 1f0))
    fake_loss = mean(binarycrossentropy.(fake_output, 0f0))
    return (real_loss + fake_loss)/2
end

generator_loss(fake_output) = mean(binarycrossentropy.(fake_output, 1f0))

function pixelwise_loss(real_output, decoded_output)
    return mean(Flux.mae.(real_output, decoded_output))
end

function train_discr(discr, fake_data, opt_discr, hparams)
    noise = randn!(similar(fake_data, (hparams.latent_dim, hparams.batch_size))) |> gpu
    ps = Flux.params(discr)
    loss, back = Zygote.pullback(ps) do
                      discr_loss(discr(noise), discr(fake_data))
    end
    grads = back(1f0)
    update!(opt_discr, ps, grads)
    return loss
end
Zygote.@nograd train_discr

function train_gan(encoder,decoder, discr, original_data, opt_en_dec, opt_discr, hparams)
    loss = Dict()
    ps = Flux.params(encoder.m, encoder.mu, encoder.logvar, decoder)
    loss["gen"], back = Zygote.pullback(ps) do
                          en_m = encoder.m(original_data)
                          encoded_img = reparameterization(encoder.mu(en_m), encoder.logvar(en_m), hparams)
                          loss["discr"] = train_discr(discr, encoded_img, opt_discr, hparams)
                          decoded_img = decoder(encoded_img)
                          0.001f0*generator_loss(discr(encoded_img)) + 0.999f0*pixelwise_loss(original_data, decoded_img)
    end
    grads = back(1f0)
    update!(opt_en_dec, ps, grads)
    return loss
end

function create_output_image(dec, fixed_noise, hparams)
    @eval Flux.istraining() = false
    fake_images = @. cpu(dec(fixed_noise))
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
    enc =  encoder(hparams)
    
    dec = decoder(hparams) 
    
    # Optimizers
    opt_dscr = ADAM(hparams.lr_dscr)
    opt_en_dec = ADAM(hparams.lr_gen)

    isdir("output")||mkdir("output")

    # Training
    train_steps = 0
    for ep in 1:hparams.epochs
        @info "Epoch $ep"
        for x in data
            # Update discriminator and generator
            loss = train_gan(enc, dec, dscr, x, opt_en_dec, opt_dscr, hparams)

            if train_steps % hparams.verbose_freq == 0
                @info("Train step $(train_steps), Discriminator loss = $(loss["discr"]), Generator loss = $(loss["gen"])")
                # Save generated fake image
                output_image = create_output_image(dec, fixed_noise , hparams)
                save(@sprintf("output/aae_steps_%06d.png", train_steps), output_image)
            end
            train_steps += 1
        end
    end

    output_image = create_output_image(dec, fixed_noise, hparams)
    save(@sprintf("output/aae_steps_%06d.png", train_steps), output_image)
end    

cd(@__DIR__)
train()
