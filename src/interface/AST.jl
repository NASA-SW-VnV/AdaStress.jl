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

Constructor for ASTMDP object. Infers various properties of MDP. Allows kwargs to be
specified flatly and automatically assigns to correct level.
"""
function ASTMDP(sim::AbstractSimulation; kwargs...)
    reset!(sim)
    s_type = infer_state(sim)
    a_type = sim isa BlackBox ? SeedAction : SampleAction
    env_info = sim isa BlackBox ? EnvironmentInfo() : infer_info(environment(sim))
    mdp = ASTMDP{s_type, a_type}(; sim=sim, env_info=env_info)

    # automatically applies kwarg to reward if match is found
    for k in keys(kwargs)
        obj = k in fieldnames(ASTMDP) || !(k in fieldnames(Reward)) ? mdp : mdp.reward
        setproperty!(obj, k, kwargs[k])
    end

    mdp.reward.heuristic = mdp.episodic ? FinalHeuristic() : mdp.reward.heuristic
    RNG_TEMP[] = mdp.rng == Random.default_rng() ? Random.Xoshiro() : deepcopy(mdp.rng)
    return mdp
end

"""
Wrap action for internal handling.
"""
convert_a(mdp::ASTMDP{<:State, SampleAction}, action::AbstractVector{<:Real}) = SampleAction(unflatten(mdp, action))
convert_a(::ASTMDP{<:State, SampleAction}, action::EnvironmentValue) = SampleAction(action)
convert_a(::ASTMDP{<:State, SeedAction}, action::UInt32) = SeedAction(action)
