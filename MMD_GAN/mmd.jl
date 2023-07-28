using LinearAlgebra

const min_var_est = 1e-8

# Helper function to compute linear time MMD with a linear kernel
function linear_mmd2(fₓ, fᵧ)
    Δ = fₓ - fᵧ
    loss = mean(sum(Δ[1:end-1, :] .* Δ[2:end, :], dims=2))
    return loss
end

# Helper function to compute linear time MMD with a polynomial kernel
function poly_mmd2(fₓ, fᵧ; d=2, α=1.0, c=2.0)
    Kₓₓ = α * sum(fₓ[1:end-1, :] .* fₓ[2:end, :], dims=2) .+ c
    Kᵧᵧ = α * sum(fᵧ[1:end-1, :] .* fᵧ[2:end, :], dims=2) .+ c
    Kₓᵧ = α * sum(fₓ[1:end-1, :] .* fᵧ[2:end, :], dims=2) .+ c
    Kᵧₓ = α * sum(fᵧ[1:end-1, :] .* fₓ[2:end, :], dims=2) .+ c

    K̃ₓₓ = mean(Kₓₓ .^ d)
    K̃ᵧᵧ = mean(Kᵧᵧ .^ d)
    K̃ₓᵧ = mean(Kₓᵧ .^ d)
    K̃ᵧₓ = mean(Kᵧₓ .^ d)

    return K̃ₓₓ + Kᵧᵧ - Kₓᵧ - Kᵧₓ
end

# Helper function to compute mixed radial basis function kernel
function _mix_rbf_kernel(X, Y, sigma_list)
    m = size(X, 1)

    Z = vcat(X, Y)
    ZZₜ = Z * Z'
    diag_ZZₜ = diag(ZZₜ)
    Z_norm_sqr = broadcast(+, diag_ZZₜ, zeros(size(ZZₜ)))
    exponent = Z_norm_sqr .- 2 * ZZₜ .+ Z_norm_sqr'

    K = zeros(size(exponent))
    for σ in sigma_list
        γ = 1.0 / (2 * σ^2)
        K += exp.(-γ * exponent)
    end

    return K[1:m, 1:m], K[1:m, m+1:end], K[m+1:end, m+1:end], length(sigma_list)
end

# Mixed Radial Basis Function Maximum Mean Discrepancy
function mix_rbf_mmd2(X, Y, sigma_list, biased=true)
    @assert size(X, 1) == size(Y, 1) "X and Y must have the same number of rows"
    Kₓₓ, Kₓᵧ, Kᵧᵧ, d = _mix_rbf_kernel(X, Y, sigma_list)
    return _mmd2(Kₓₓ, Kₓᵧ, Kᵧᵧ, false, biased)
end

function mix_rbf_mmd2_and_ratio(X, Y, sigma_list, biased=true)
    Kₓₓ, Kₓᵧ, Kᵧᵧ, d = _mix_rbf_kernel(X, Y, sigma_list)
    return _mmd2_and_ratio(Kₓₓ, Kₓᵧ, Kᵧᵧ, false, biased)
end

# Helper function to compute variance based on kernel matrices
function _mmd2(Kₓₓ, Kₓᵧ, Kᵧᵧ, const_diagonal=false, biased=false)
    m = size(Kₓₓ, 1)

    # Get the various sums of kernels that we'll use
    if const_diagonal !== false
        diagₓ = diagᵧ = const_diagonal
        sum_diagₓ = sum_diagᵧ = m * const_diagonal
    else
        diagₓ = diag(Kₓₓ)
        diagᵧ = diag(Kᵧᵧ)
        sum_diagₓ = sum(diagₓ)
        sum_diagᵧ = sum(diagᵧ)
    end

    Kₜₓₓ_sums = sum(Kₓₓ, dims=2) .- diagₓ
    Kₜᵧᵧ_sums = sum(Kᵧᵧ, dims=2) .- diagᵧ
    Kₓᵧ_sums₁ = sum(Kₓᵧ, dims=1)

    Kₜₓₓ_sum = sum(Kₜₓₓ_sums)
    Kₜᵧᵧ_sum = sum(Kₜᵧᵧ_sums)
    Kₓᵧ_sum = sum(Kₓᵧ_sums₁)

    if biased
        mmd2 = ((Kₜₓₓ_sum + sum_diagₓ) / (m * m)
                +
                (Kₜᵧᵧ_sum + sum_diagᵧ) / (m * m)
                -
                2.0 * Kₓᵧ_sum / (m * m))
    else
        mmd2 = (Kₜₓₓ_sum / (m * (m - 1))
                +
                Kₜᵧᵧ_sum / (m * (m - 1))
                -
                2.0 * Kₓᵧ_sum / (m * m))
    end

    return mmd2
end

function mmd2_and_ratio(K_XX, K_XY, K_YY; const_diagonal::Bool=false, biased::Bool=false)
    mmd2, var_est = mmd2_and_variance(Kₓₓ, Kₓᵧ, Kᵧᵧ, const_diagonal=const_diagonal, biased=biased)
    loss = mmd2 / √(max(var_est, min_var_est))
    return loss, mmd2, var_est
end

function _mmd2_and_variance(Kₓₓ, Kₓᵧ, Kᵧᵧ; const_diagonal=false, biased=false)
    m = size(Kₓₓ, 1)    # assume X, Y are the same shape

    # Get the various sums of kernels that we'll use
    if const_diagonal !== false
        diagₓ = diagᵧ = const_diagonal
        sum_diagₓ = sum_diagᵧ = m * const_diagonal
        sum_diag2ₓ = sum_diag2ᵧ = m * const_diagonal^2
    else
        diagₓ = diagm(Kₓₓ)                       # (m,)
        diagᵧ = diagm(Kᵧᵧ)                       # (m,)
        sum_diagₓ = sum(diagₓ)
        sum_diagᵧ = sum(diagᵧ)
        sum_diag2ₓ = dot(diagₓ, diagₓ)
        sum_diag2ᵧ = dot(diagᵧ, diagᵧ)
    end

    Kₜₓₓ_sums = sum(Kₓₓ, dims=2) .- diagₓ        # \tilde{K}_XX * e = Kₓₓ * e - diagₓ
    Kₜᵧᵧ_sums = sum(Kᵧᵧ, dims=2) .- diagᵧ        # \tilde{K}_YY * e = Kᵧᵧ * e - diagᵧ
    Kₓᵧ_sums₁ = sum(Kₓᵧ, dims=1)                # K_{XY}^T * e
    Kₓᵧ_sums₂ = sum(Kₓᵧ, dims=2)                # K_{XY} * e

    Kₜₓₓ_sum = sum(Kₜₓₓ_sums)
    Kₜᵧᵧ_sum = sum(Kₜᵧᵧ_sums)
    Kₓᵧ_sum = sum(Kₓᵧ_sums₁)

    Kₜₓₓ_2_sum = sum(Kₓₓ .^ 2) .- sum_diag2ₓ     # \| \tilde{K}_XX \|_F^2
    Kₜᵧᵧ_2_sum = sum(Kᵧᵧ .^ 2) .- sum_diag2ᵧ     # \| \tilde{K}_YY \|_F^2
    Kₓᵧ_2_sum = sum(Kₓᵧ .^ 2)                   # \| K_{XY} \|_F^2

    if biased
        mmd2 = ((Kₜₓₓ_sum + sum_diagₓ) / (m * m)
                +
                (Kₜᵧᵧ_sum + sum_diagᵧ) / (m * m)
                -
                2.0 * Kₓᵧ_sum / (m * m))
    else
        mmd2 = (Kₜₓₓ_sum / (m * (m - 1))
                +
                Kₜᵧᵧ_sum / (m * (m - 1))
                -
                2.0 * Kₓᵧ_sum / (m * m))
    end

    var_est = (
        2.0 / (m^2 * (m - 1.0)^2) * (2 * dot(Kₜₓₓ_sums, Kₜₓₓ_sums) - Kₜₓₓ_2_sum + 2 * dot(Kₜᵧᵧ_sums, Kₜᵧᵧ_sums) - Kₜᵧᵧ_2_sum)
        -
        (4.0 * m - 6.0) / (m^3 * (m - 1.0)^3) * (Kₜₓₓ_sum^2 + Kₜᵧᵧ_sum^2)
        +
        4.0 * (m - 2.0) / (m^3 * (m - 1.0)^2) * (dot(Kₓᵧ_sums₂, Kₓᵧ_sums₂) + dot(Kₓᵧ_sums₁, Kₓᵧ_sums₁))
        -
        4.0 * (m - 3.0) / (m^3 * (m - 1.0)^2) * (Kₓᵧ_2_sum) - (8 * m - 12) / (m^5 * (m - 1)) * Kₓᵧ_sum^2
        +
        8.0 / (m^3 * (m - 1.0)) * (
            1.0 / m * (Kₜₓₓ_sum + Kₜᵧᵧ_sum) * Kₓᵧ_sum
            -
            dot(Kₜₓₓ_sums, Kₓᵧ_sums₂)
            -
            dot(Kₜᵧᵧ_sums, Kₓᵧ_sums₁))
    )

    return mmd2, var_est
end
