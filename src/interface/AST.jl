"""
Samples environment, returning EnvironmentValue or array (default).
"""
function Base.rand(env::Environment; flat::Bool=true)
	value = EnvironmentValue(k => rand(dist) for (k, dist) in env)
	return flat ? flatten(env, value) : value
end

"""
Returns action type of ASTMDP.
"""
actiontype(::ASTMDP{A}) where A = A

"""
Infers dimension of action space.
"""
act_dim(mdp::ASTMDP{SampleAction}) = sum(info.n for info in values(mdp.env_info))

"""
Infers dimension of state space.
"""
obs_dim(mdp::ASTMDP) = length(observe(mdp.sim))

"""
Flattens EnvironmentValue into single array.
#TODO: pre-allocate array?
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
function unflatten(mdp::ASTMDP{SampleAction}, action::Vector{<:Real})
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
Calculates log probability of sample from environment variable.
"""
function logprob(distribution::Any, value::Any, marginalize::Bool)
    logp = logpdf(distribution, value)
    if marginalize
        logp -= logpdf(distribution, mode(distribution))
    end
    return logp
end

"""
Calculates total log probability of environment value.
"""
function logprob(env::Environment, value::EnvironmentValue, marginalize::Bool)
    return sum(logprob(env[k], value[k], marginalize) for k in keys(env))
end

function get_info(env::Environment)
    env_info = EnvironmentInfo()
	for (k, dist) in env
		sample = rand(dist)
		array = flatten(dist, sample)
		env_info[k] = VariableInfo(length(array), typeof(sample))
	end
    return env_info
end

"""
Constructor for ASTMDP object. Infers various properties of MDP.
"""
function ASTMDP(sim::GrayBox; kwargs...)
    reset!(sim)
	return ASTMDP{SampleAction}(; sim=sim, kwargs..., env_info=get_info(environment(sim)))
end

function ASTMDP(sim::BlackBox; kwargs...)
    reset!(sim)
	return ASTMDP{SeedAction}(; sim=sim, kwargs...)
end
