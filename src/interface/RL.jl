"""
Reset MDP.
"""
function reset!(mdp::ASTMDP)
    reset!(mdp.sim)
    reset!(mdp.reward.heuristic)
    return nothing
end

"""
Return set of available actions as sampleable object.
"""
actions(mdp::ASTMDP{<:State, SampleAction}) = environment(mdp.sim)
actions(::ASTMDP{<:State, SeedAction}) = UInt32

"""
Return observation of environment.
"""
observe(mdp::ASTMDP{ObservableState, <:Action}) = Float32.(observe(mdp.sim))

"""
Step simulation and return log probability of transition.
"""
function evaluate!(mdp::ASTMDP{<:State, A}, a::A) where A <: SampleAction
    env = environment(mdp.sim)
    val = a.sample
    logp = logprob(env, val, mdp.reward.marginalize)
    step!(mdp.sim, val)
    return logp
end

function evaluate!(mdp::ASTMDP{<:State, A}, a::A) where A <: SeedAction
    copy!(RNG_TEMP, mdp.rng)
    Random.seed!(mdp.rng, a.seed)
    logp = step!(mdp.rng, mdp.sim)
    copy!(mdp.rng, RNG_TEMP)
    return logp
end

"""
Apply raw action to environment and return reward.
"""
act!(mdp::ASTMDP{<:State, <:Action}, action) = act!(mdp, convert_a(mdp, action))

"""
Apply converted action to environment and return reward.
"""
function act!(mdp::ASTMDP{<:State, A}, a::A) where A <: Action
    # rewards (partial application)
    event = mdp.episodic ? isterminal(mdp.sim) && isevent(mdp.sim) : isevent(mdp.sim)
    bonus = event ? mdp.reward.event_bonus : 0.0
    heur = mdp.reward.heuristic(mdp)
    rew = custom_reward(mdp.sim, a)

    # stepping and likelihood evaluation
    logp = evaluate!(mdp, a)

    # final reward calculation
    return mdp.reward.reward_function(logp, bonus, heur(mdp)) + rew(mdp.sim)
end

"""
Return Boolean indicating termination status.
"""
terminated(mdp::ASTMDP) = isterminal(mdp.sim) || !mdp.episodic && isevent(mdp.sim)

# Connection between AdaStress internal functions and CommonRLInterface

CommonRLInterface.reset!(mdp::AbstractASTMDP) = reset!(mdp)

CommonRLInterface.actions(mdp::AbstractASTMDP) = actions(mdp)

CommonRLInterface.observe(mdp::AbstractASTMDP) = observe(mdp)

CommonRLInterface.act!(mdp::AbstractASTMDP, action) = act!(mdp, action)

CommonRLInterface.terminated(mdp::AbstractASTMDP) = terminated(mdp)
