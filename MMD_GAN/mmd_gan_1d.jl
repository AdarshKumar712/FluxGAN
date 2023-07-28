using Base.Iterators: partition
using Flux
using Flux.Optimise: update!
using Flux: logitbinarycrossentropy, binarycrossentropy
using Statistics
using Parameters: @with_kw
using Random
using Printf
using CUDA
using Zygote
using Distributions

include("./mmd.jl")

@with_kw struct HyperParams
    data_size::Int = 10000
    batch_size::Int = 128
    latent_dim::Int = 1
    epochs::Int = 1000
    verbose_freq::Int = 1000
    num_gen::Int = 1
    num_enc_dec::Int = 5
    lr_enc::Float64 = 1.0e-4
    lr_dec::Float64 = 1.0e-4
    lr_gen::Float64 = 1.0e-4

    lambda_AE::Float64 = 8.0
    target_param::Tuple{Float64,Float64} = (23.0, 1.0)
    noise_param::Tuple{Float64,Float64} = (0.0, 1.0)
    base::Float64 = 1.0
    sigma_list::Array{Float64,1} = [1.0, 2.0, 4.0, 8.0, 16.0] ./ base
end

function generator()
    return Chain(
        Dense(1, 7),
        elu,
        Dense(7, 13),
        elu,
        Dense(13, 7),
        elu,
        Dense(7, 1)
    )
end

function encoder()
    return Chain(Dense(1, 11), elu, Dense(11, 29), elu)
end

function decoder()
    return Chain(Dense(29, 11), elu, Dense(11, 1))
end

function data_sampler(hparams, target)
    return rand(Normal(target[1], target[2]), (hparams.batch_size, 1))
end

# Initialize models and optimizers
function train()
    hparams = HyperParams()
    mse = Flux.mse

    gen = generator()
    enc = encoder()
    dec = decoder()

    # Optimizers
    gen_opt = ADAM(hparams.lr_gen)
    enc_opt = ADAM(hparams.lr_enc)
    dec_opt = ADAM(hparams.lr_dec)

    cum_dis_loss = 0.0
    cum_gen_loss = 0.0

    # Training
    losses_gen = []
    losses_dscr = []
    train_steps = 0
    # Training loop
    gen_ps = Flux.params(gen)
    enc_ps = Flux.params(enc)
    dec_ps = Flux.params(dec)
    @showprogress for ep in 1:hparams.epochs
        for _ in 1:hparams.num_enc_dec
            loss, back = Zygote.pullback(Flux.params(enc, dec)) do
                target = data_sampler(hparams, hparams.target_param)
                noise = data_sampler(hparams, hparams.noise_param)
                encoded_target = enc(target')
                decoded_target = dec(encoded_target)
                L2_AE_target = Flux.mse(decoded_target, target)
                transformed_noise = gen(noise')
                encoded_noise = enc(transformed_noise)
                decoded_noise = dec(encoded_noise)
                L2_AE_noise = Flux.mse(decoded_noise, transformed_noise)
                MMD = mix_rbf_mmd2(encoded_target, encoded_noise, hparams.sigma_list)
                MMD = relu(MMD)
                L_MMD_AE = -1.0 * (sqrt(MMD) - hparams.lambda_AE * (L2_AE_noise + L2_AE_target))
            end
            grads = back(1.0f0)
            update!(enc_opt, enc_ps, grads)
            update!(dec_opt, dec_ps, grads)
            push!(losses_dscr, loss)
        end
        for _ in 1:hparams.num_gen
            loss, back = Zygote.pullback(gen_ps) do
                target = data_sampler(hparams, hparams.target_param)
                noise = data_sampler(hparams, hparams.noise_param)
                encoded_target = enc(target')
                encoded_noise = enc(gen(noise'))
                MMD = sqrt(relu(mix_rbf_mmd2(encoded_target, encoded_noise, hparams.sigma_list)))
            end
            grads = back(1.0f0)
            update!(gen_opt, gen_ps, grads)
            push!(losses_gen, loss)
        end
    end
end

function plot_results(n_samples, range)
    target = [data_sampler(hparams, hparams.target_param) for _ in 1:n_samples]
    target = collect(Iterators.flatten(target))
    transformed_noise = [gen(data_sampler(hparams, hparams.noise_param)')' for _ in 1:n_samples]
    transformed_noise = collect(Iterators.flatten(transformed_noise))
    histogram(target, bins=range)
    histogram!(transformed_noise, bins=range)
end
