"""
Sample environment, returning `EnvironmentValue` (default) or array.
"""
function Base.rand(env::Environment; flat::Bool=false)
	value = EnvironmentValue(k => rand(dist) for (k, dist) in env)
	return flat ? flatten(env, value) : value
end

"""
Infer dimension of action space.
"""
act_dim(mdp::ASTMDP{<:State, SampleAction}) = sum(info.n for info in values(mdp.env_info))

"""
Infer dimension of state space.
"""
obs_dim(mdp::ASTMDP{ObservableState, <:Action}) = length(observe(mdp.sim))

"""
Return ordered list dictionary keys.
"""
orderedkeys(dict::Dict) = sort!(collect(keys(dict)))

"""
Flatten `EnvironmentValue` into array.
"""
function flatten(env::Environment, value::EnvironmentValue)
	return reduce(append!, flatten(env[k], value[k]) for k in orderedkeys(env))
end

"""
Reconstruct `EnvironmentValue` from array.
"""
function unflatten(mdp::ASTMDP{<:State, SampleAction}, action::AbstractVector{<:Real})
	value = EnvironmentValue()
	env = environment(mdp.sim)

	i = 0
	for k in orderedkeys(env)
		n = mdp.env_info[k].n
		array = action[i+1:i+n]
		value[k] = unflatten(env[k], array)
		i += n
	end
	return value
end

"""
Calculate log probability of sample from environment variable.
"""
function logprob(distribution::Any, value::Any, marginalize::Bool)
    logp = logpdf(distribution, value)
    if marginalize
        logp -= logpdf(distribution, mode(distribution))
    end
    return logp
end

"""
Calculate total log probability of environment value.
"""
function logprob(env::Environment, value::EnvironmentValue, marginalize::Bool)
    return sum(logprob(env[k], value[k], marginalize) for k in keys(env))
end

"""
Infer information about simulation environment.
"""
function infer_info(env::Environment)
    env_info = EnvironmentInfo()
	for (k, dist) in env
		sample = rand(dist)
		array = flatten(dist, sample)
		env_info[k] = VariableInfo(length(array), typeof(sample))
	end
    return env_info
end

"""
Infer type of state.
"""
function infer_state(sim::AbstractSimulation)
    try
        isempty(observe(sim)) ? UnobservableState : ObservableState
    catch e
        e isa UnimplementedError ? UnobservableState : throw(e)
    end
end

"""
    ASTMDP(sim::AbstractSimulation; kwargs...)

Constructor for ASTMDP object. Infers various properties of MDP.
"""
function ASTMDP(sim::AbstractSimulation; kwargs...)
    reset!(sim)
    act_type = sim isa BlackBox ? SeedAction : SampleAction
    env_info = sim isa BlackBox ? EnvironmentInfo() : infer_info(environment(sim))
    mdp = ASTMDP{infer_state(sim), act_type}(; sim=sim, kwargs..., env_info=env_info)
    mdp.reward.heuristic = mdp.episodic ? FinalHeuristic() : mdp.reward.heuristic
    global RNG_TEMP = deepcopy(mdp.rng)
    return mdp
end

"""
Wrap action for internal handling.
"""
convert_a(mdp::ASTMDP{<:State, SampleAction}, action::AbstractVector{<:Real}) = SampleAction(unflatten(mdp, action))
convert_a(::ASTMDP{<:State, SampleAction}, action::EnvironmentValue) = SampleAction(action)
convert_a(::ASTMDP{<:State, SeedAction}, action::UInt32) = SeedAction(action)
