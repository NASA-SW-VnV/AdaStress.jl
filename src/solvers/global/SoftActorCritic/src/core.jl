# ******************************************************************************************
# Notices:
#
# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.
#
# Disclaimers
#
# No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF ANY KIND,
# EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY
# THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR FREEDOM FROM INFRINGEMENT, ANY
# WARRANTY THAT THE SUBJECT SOFTWARE WILL BE ERROR FREE, OR ANY WARRANTY THAT DOCUMENTATION,
# IF PROVIDED, WILL CONFORM TO THE SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT, IN ANY MANNER,
# CONSTITUTE AN ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS,
# RESULTING DESIGNS, HARDWARE, SOFTWARE PRODUCTS OR ANY OTHER APPLICATIONS RESULTING FROM
# USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY DISCLAIMS ALL WARRANTIES AND
# LIABILITIES REGARDING THIRD-PARTY SOFTWARE, IF PRESENT IN THE ORIGINAL SOFTWARE, AND
# DISTRIBUTES IT "AS IS."
#
# Waiver and Indemnity:  RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE UNITED
# STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT.
# IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS IN ANY LIABILITIES, DEMANDS, DAMAGES,
# EXPENSES OR LOSSES ARISING FROM SUCH USE, INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON,
# OR RESULTING FROM, RECIPIENT'S USE OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND
# HOLD HARMLESS THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL
# AS ANY PRIOR RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR ANY
# SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS AGREEMENT.
# ******************************************************************************************

const LOG_STD_MAX = 2
const LOG_STD_MIN = -20

"""
Create multilayer perceptron with given parameters.
"""
function mlp(sizes::Vector{Int}, activation::Function, output_activation::Function=identity)
    layers = []
    for j = 1:length(sizes) - 1
        act = j < length(sizes) - 1 ? activation : output_activation
        push!(layers, Dense(sizes[j], sizes[j + 1], act))
    end
    return Chain(layers...) |> dev
end

"""
Gaussian MLP policy.
"""
mutable struct SquashedGaussianMLPActor
    net::Chain                              # primary network
    mu_layer::Dense                         # mean layer
    log_std_layer::Dense                    # log standard deviation layer
    act_mins::AbstractVector{Float32}       # minimum values of actions
    act_maxs::AbstractVector{Float32}       # maximum values of actions
    rng::AbstractRNG                        # internal RNG
    rng_gpu::Union{AbstractRNG, Nothing}    # internal RNG for GPU calculations
    linearized::Bool                        # linearized squashing (allows post-analysis)
end

function SquashedGaussianMLPActor(
	obs_dim::Int,
	act_dim::Int,
	hidden_sizes::Vector{Int},
	activation::Function,
	act_mins::Vector{Float64},
	act_maxs::Vector{Float64},
	rng::AbstractRNG,
    linearized::Bool
)
    net = mlp(vcat(obs_dim, hidden_sizes), activation, activation) |> dev
    mu_layer = Dense(hidden_sizes[end], act_dim) |> dev
    log_std_layer = Dense(hidden_sizes[end], act_dim) |> dev
    act_mins, act_maxs = dev(Float32.(act_mins)), dev(Float32.(act_maxs))
    if WITH_GPU[]
        rng_gpu = CURAND.RNG()
        seed = rand(rng, UInt32)
        Random.seed!(rng_gpu, seed)
    else
        rng_gpu = nothing
    end
    return SquashedGaussianMLPActor(net, mu_layer, log_std_layer, act_mins, act_maxs, rng, rng_gpu, linearized)
end

"""
Calculate log pdf of normal distribution.
"""
function normal_logpdf(μ::AbstractMatrix{Float32}, σ::AbstractMatrix{Float32}, x::AbstractMatrix{Float32})
    lz = sum(-0.5f0 .* ((x .- μ) ./ σ).^2; dims=1)
    lden = (size(μ, 1) * Float32(log(2π)) / 2) .+ sum(log.(σ); dims=1)
    lpdf = lz .- lden
    lpdf = dropdims(lpdf; dims=1)
    return lpdf
end

"""
Calculate random normal deviates on GPU (in-place).
"""
function normal_deviates(rng::AbstractRNG, sz::NTuple{N, Int64}) where N
    z = CuArray{Float32}(undef, sz)
    randn!(rng, z)
    return z
end
Zygote.@nograd normal_deviates

"""
Retrieve action (and optional log probability) from policy.
"""
function (pi::SquashedGaussianMLPActor)(
			obs::AbstractArray{Float32},
			deterministic::Bool=false,
			with_logprob::Bool=true
)
    net_out = pi.net(obs)
    mu = pi.mu_layer(net_out)
    log_std = pi.log_std_layer(net_out)
    log_std = clamp.(log_std, LOG_STD_MIN, LOG_STD_MAX)
    std = exp.(log_std)

    # Pre-squash distribution and sample
    if deterministic
        pi_action = mu
    else
        if mu isa CuArray
            z = normal_deviates(pi.rng_gpu, size(mu))
        else
            z = randn(pi.rng, Float32, size(mu))
        end
        pi_action = mu .+ std .* Zygote.dropgrad(z)
    end

    if with_logprob
        logp_pi = normal_logpdf(mu, std, pi_action)
        logp_pi = logp_pi .- dropdims(sum((2.0f0 .* (log(2.0f0) .- pi_action .- softplus.(-2.0f0 .* pi_action))); dims=1); dims=1)
    else
        logp_pi = NaN
    end

    if pi.linearized # probabilities still correspond to non-linear squashing
        pi_action = clamp.(pi_action, pi.act_mins, pi.act_maxs)
    else
        pi_action = tanh.(pi_action)
        pi_action = @. pi.act_mins + (pi.act_maxs - pi.act_mins) * (pi_action / 2 + 0.5f0)
    end

    return pi_action, logp_pi
end

"""
Q-value function.
"""
mutable struct MLPQFunction
    q::Chain
end

function MLPQFunction(obs_dim::Int, act_dim::Int, hidden_sizes::Vector{Int}, activation::Function)
    q = mlp(vcat(obs_dim + act_dim, hidden_sizes, 1), activation)
    return MLPQFunction(q)
end

"""
Determine Q-value of observation and action.
"""
function (qf::MLPQFunction)(obs::AbstractMatrix{Float32}, act::AbstractMatrix{Float32})
    q = qf.q(cat(obs, act; dims=1))
    q = dropdims(q; dims=1)
    return q
end

"""
Actor-critic agent.
"""
mutable struct MLPActorCritic <: GlobalResult
    pi::SquashedGaussianMLPActor
    qs::Vector{MLPQFunction}
end

function MLPActorCritic(
	obs_dim::Int,
	act_dim::Int,
	act_mins::Vector{Float64},
	act_maxs::Vector{Float64},
	hidden_sizes::Vector{Int}=[100,100,100],
	num_q::Int=2,
	activation::Function=SoftActorCritic.relu,
	rng::AbstractRNG=Random.GLOBAL_RNG,
    linearized::Bool=false
)
    pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_mins, act_maxs, rng, linearized)
    qs = [MLPQFunction(obs_dim, act_dim, hidden_sizes, activation) for _ in 1:num_q]
    return MLPActorCritic(pi, qs)
end

"""
Retrieve action from policy.
"""
function (ac::MLPActorCritic)(obs::AbstractVector{Float32}, deterministic::Bool=true)
    a, _ = ac.pi(obs, deterministic, false)
    return a
end

"""
Define native softplus function to avoid CUDA bugs. #TODO: removable?
"""
softplus(x::Real) = x > 0 ? x + log(1 + exp(-x)) : log(1 + exp(x))

"""
Define native relu function to avoid Flux bugs. #TODO: removable?
"""
relu(x::Real) = max(x, 0)
