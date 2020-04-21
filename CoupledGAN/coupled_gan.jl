using Base.Iterators: partition
using Flux
using Flux.Optimise: update!
using Flux: binarycrossentropy, mse, logitbinarycrossentropy
using Images
using MLDatasets
using Statistics, StatsBase
using Parameters: @with_kw
using Random
using Printf
using CUDAapi
using Zygote
using Augmentor
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
    verbose_freq::Int = 500
    output_x::Int = 4        # No. of sample images to concatenate along x-axis 
    output_y::Int = 3        # No. of sample images to concatenate along y-axis
    lr_dscr::Float64 = 0.0001
    lr_gen::Float64 = 0.0001
end


function load_data(hparams)
    # Load MNIST dataset
    images, labels = MLDatasets.MNIST.traindata(Float32)
    # Normalize to [-1, 1] and convert it to WHCN
    image_tensor = permutedims(reshape(@.(2f0 * images - 1f0), 28, 28, 1, :), (2, 1, 3, 4))
    img_rotated = similar(image_tensor)
    n = size(image_tensor,4)
    for i in 1:n
        img_rotated[:,:,1,i] = augment(image_tensor[:,:,1,i], Rotate90())
    end 
    img_tensor = image_tensor[:,:,:,1:Int(n/2)]
    img_rotated = img_rotated[:,:,:,Int(n/2)+1:end]
    img_tensor = reshape(img_tensor, :, size(img_tensor,4))
    img_rotated = reshape(img_rotated, :, size(img_rotated,4))
    data = [(img_tensor[:,r], img_rotated[:,r]) |> gpu for r in partition(1:30000, hparams.batch_size)]
    return data
end

mutable struct gan_model
    m_common
    m1
    m2
end

function generator(args)
    m_common = Chain(Dense(args.latent_dim, 256),x-> leakyrelu.(x,0.2f0), BatchNorm(256), Dense(256, 512), x-> leakyrelu.(x,0.2f0), BatchNorm(512)) |> gpu
    m1 = Chain(Dense(512, 1024),x-> leakyrelu.(x,0.2f0), BatchNorm(1024),Dense(1024, 784, tanh)) |> gpu
    m2 = Chain(Dense(512, 1024),x-> leakyrelu.(x,0.2f0), BatchNorm(1024),Dense(1024, 784, tanh)) |> gpu
    gan_model(m_common,m1, m2)
end

function discriminator(args) 
    m1 = Chain(Dense(784, 512),x-> leakyrelu.(x,0.2f0), Dense(512, 256),x-> leakyrelu.(x,0.2f0)) |> gpu
    m2 = Chain(Dense(784, 512),x-> leakyrelu.(x,0.2f0), Dense(512, 256),x-> leakyrelu.(x,0.2f0)) |> gpu
    m_common = Chain(Dense(256,1, sigmoid)) |> gpu
    gan_model(m_common, m1, m2)
end

# Loss functions
function discr_loss(real_output, fake_output)
    real_loss = mean(binarycrossentropy.(real_output, 1f0))
    fake_loss = mean(binarycrossentropy.(fake_output, 0f0))
    return real_loss + fake_loss
end

generator_loss(fake_output) = mean(binarycrossentropy.(fake_output, 1f0))

function train_discr(discr, img1, img2, fake_img1, fake_img2, opt_discr)
    ps_d = Flux.params(discr.m_common, discr.m1, discr.m2)
    loss = 0.0
    loss, back = Zygote.pullback(ps_d) do
                    x1_real = discr.m_common(discr.m1(img1))
                    x2_real = discr.m_common(discr.m2(img2))
                    x1_fake = discr.m_common(discr.m1(fake_img1))
                    x2_fake = discr.m_common(discr.m2(fake_img2))
                    discr_loss(x1_real, x1_fake) + discr_loss(x2_real, x2_fake)
    end
    grads = back(1f0)
    for i in params(discr.m_common)
        grads.grads[i] ./=2
    end
    update!(opt_discr, ps_d, grads)
    return loss                      
end

Zygote.@nograd train_discr 

function train_gan(gen, discr, img1, img2, opt_gen, opt_discr, hparams)
    noise = randn!(similar(img1, (hparams.latent_dim, hparams.batch_size))) |> gpu
    ps = Flux.params(gen.m_common, gen.m1, gen.m2)
    loss = Dict()
    loss["gen"], back = Zygote.pullback(ps) do
                            x = gen.m_common(noise)
                            fake_1 = gen.m1(x) 
                            fake_2 = gen.m2(x)
                            loss["discr"] = train_discr(discr, img1, img2, fake_1, fake_2, opt_discr) 
                            generator_loss(discr.m_common(discr.m1(fake_1))) + generator_loss(discr.m_common(discr.m2(fake_2)))
    end
    grads = back(1f0)
    for i in params(gen.m_common)
         grads.grads[i] ./=2
    end
    update!(opt_gen, ps, grads)
    return loss
end

function create_output_image(gen, fixed_noise, hparams)
    @eval Flux.istraining() = false
    fake_i = @. cpu(gen.m1(gen.m_common(fixed_noise)))
    fake_i = @. reshape(fake_i, 28,28,1,size(fake_i,2))
    fake_i_2 = @. cpu(gen.m2(gen.m_common(fixed_noise)))
    fake_i_2 = @. reshape(fake_i_2, 28,28,1,size(fake_i_2,2))
    fake_images =@. hcat(fake_i, fake_i_2)
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
    opt_dscr = ADAM(hparams.lr_dscr, (0.5,0.99))
    opt_gen = ADAM(hparams.lr_gen, (0.5,0.99))

    isdir("output")||mkdir("output")

    # Training
    train_steps = 0
    for ep in 1:hparams.epochs
        @info "Epoch $ep"
        for (img1, img2) in data
            loss = train_gan(gen, dscr, img1, img2, opt_gen, opt_dscr, hparams)
            
            if train_steps % hparams.verbose_freq == 0
                @info("Train step $(train_steps), Discriminator loss = $(loss["discr"]), Generator loss = $(loss["gen"])")
                # Save generated fake image
                output_image = create_output_image(gen, fixed_noise, hparams)
                save(@sprintf("output/cogan_steps_%06d.png", train_steps), output_image)
            end
            train_steps += 1
        end
    end
    
    output_image = create_output_image(gen, fixed_noise, hparams)
    save(@sprintf("output/cogan_steps_%06d.png", train_steps), output_image)
end

cd(@__DIR__)
train()
