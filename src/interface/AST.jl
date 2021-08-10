"""
Internal abstract type.
"""
abstract type AbstractSimulation end

"""
Stores sample value and corresponding log probability.
"""
mutable struct Sample
	value::Any
	logprob::Float64
end

"""
Stores distributions of environment variables.
"""
const Environment = Dict{Symbol, Sampleable}

"""
Stores samples of environment distributions.
"""
const EnvironmentSample = Dict{Symbol, Sample}

"""
Stores values of environment samples.
"""
const EnvironmentValue = Dict{Symbol, Any}

"""
Samples environment, returning EnvironmentValue or array (default).
"""
function Base.rand(env::Environment; flat::Bool=true)
	value = EnvironmentValue(k => rand(dist) for (k, dist) in env)
	return flat ? flatten(env, value) : value
end

"""
Computes log probability of environment sample.
"""
logprob(sample::EnvironmentSample) = sum(s.logprob for (_, s) in sample)

"""
Stores inferred properties of environment variable.
"""
struct EnvironmentInfo
	n::Int64	# environment variable dimensionality
	t::Type		# environment variable type
end

"""
MDP object for AST. Wraps simulation and contains auxiliary information and parameters.
#TODO: add hooks for annealing.
"""
@with_kw mutable struct ASTMDP <: CommonRLInterface.AbstractEnv
	sim::AbstractSimulation						        # simulation wrapping system under test
	reward_bonus::Float64=0.0   				        # bonus for reaching event
    reward_function::RewardFunction = WeightedReward()  # reward function for AST
    marginalize::Bool=true                              # use marginalized probabilities
    heuristic::DistanceHeuristic=DistanceGradient()     # distance heuristic
	env_info::Dict{Symbol, EnvironmentInfo}		        # inferred environment properties
end

"""
Infers dimension of action space.
"""
act_dim(mdp::ASTMDP) = sum(info.n for info in values(mdp.env_info))

"""
Infers dimension of state space.
"""
obs_dim(mdp::ASTMDP) = length(observe(mdp.sim))

"""
Flattens EnvironmentValue into single array.
"""
function flatten(env::Environment, value::EnvironmentValue)
	action = Float32[]
	for k in sort(collect(keys(env)))
		array = flatten(env[k], value[k])
		append!(action, array)
	end
	return action
end

"""
Reconstructs EnvironmentValue from single array.
"""
function unflatten(mdp::ASTMDP, action::Vector{<:Real})
	value = EnvironmentValue()
	env = environment(mdp.sim)

	i = 0
	for k in sort(collect(keys(env)))
		n = mdp.env_info[k].n
		array = action[i+1:i+n]
		value[k] = unflatten(env[k], array)
		i += n
	end
	return value
end

"""
Calculates marginalized log probability of sample from environment distribution.
"""
function logprob(distribution::Any, value::Any, marginalize::Bool)
    logp = logpdf(distribution, value)
    if marginalize
        logp -= logpdf(distribution, mode(distribution))
    end
    return logp
end

"""
Converts EnvironmentValue to EnvironmentSample, calculating associated log probabilities.
"""
function create_sample(env::Environment, value::EnvironmentValue, marginalize::Bool)
	sample = EnvironmentSample()
	for k in keys(env)
		dist, val = env[k], value[k]
		sample[k] = Sample(val, logprob(dist, val, marginalize))
	end
	return sample
end

"""
Calculates AST reward from MDP and sample.
"""
function reward(mdp::ASTMDP, sample::EnvironmentSample)
	logp = logprob(sample)
	event = isevent(mdp.sim)
    heuristic = (mdp.heuristic)(distance(mdp.sim))
	return reward(mdp.reward_function, logp, event, heuristic, mdp.reward_bonus)
end

"""
Constructor for ASTMDP object. Infers various properties of MDP.
"""
function ASTMDP(sim::AbstractSimulation; kwargs...)
    reset!(sim)
	env = environment(sim)
	env_info = Dict{Symbol, EnvironmentInfo}()
	for (k, dist) in env
		sample = rand(dist)
		array = flatten(dist, sample)
		env_info[k] = EnvironmentInfo(length(array), typeof(sample))
	end
	return ASTMDP(; sim=sim, kwargs..., env_info=env_info)
end
