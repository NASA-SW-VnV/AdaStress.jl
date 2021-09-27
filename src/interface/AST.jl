"""
Samples environment, returning EnvironmentValue or array (default).
"""
function Base.rand(env::Environment; flat::Bool=false)
	value = EnvironmentValue(k => rand(dist) for (k, dist) in env)
	return flat ? flatten(env, value) : value
end

"""
Infers dimension of action space.
"""
act_dim(mdp::ASTMDP{<:State, SampleAction}) = sum(info.n for info in values(mdp.env_info))

"""
Infers dimension of state space.
"""
obs_dim(mdp::ASTMDP{ObservableState, <:Action}) = length(observe(mdp.sim))

"""
Returns ordered list dictionary keys.
"""
orderedkeys(dict::Dict) = sort!(collect(keys(dict)))

"""
Flattens EnvironmentValue into single array.
"""
function flatten(env::Environment, value::EnvironmentValue)
	return reduce(append!, flatten(env[k], value[k]) for k in orderedkeys(env))
end

"""
Reconstructs EnvironmentValue from single array.
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

"""
Infers information about simulation environment.
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
Infers type of state.
"""
function infer_state(sim::AbstractSimulation)
    try
        isempty(observe(sim)) ? UnobservableState : ObservableState
    catch e
        e isa UnimplementedError ? UnobservableState : throw(e)
    end
end

"""
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

convert_a(mdp::ASTMDP{<:State, SampleAction}, action::AbstractVector{<:Real}) = SampleAction(unflatten(mdp, action))
convert_a(::ASTMDP{<:State, SampleAction}, action::EnvironmentValue) = SampleAction(action)
convert_a(::ASTMDP{<:State, SeedAction}, action::UInt32) = SeedAction(action)
