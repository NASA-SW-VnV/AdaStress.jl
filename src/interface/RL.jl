"""
Resets MDP.
"""
function reset!(mdp::ASTMDP)
    reset!(mdp.sim)
    reset!(mdp.reward.heuristic)
end

"""
Returns set of available actions as sampleable object.
"""
actions(mdp::ASTMDP{<:State, SampleAction}) = environment(mdp.sim)
actions(::ASTMDP{<:State, SeedAction}) = UInt32

"""
Returns observation of environment.
"""
observe(mdp::ASTMDP{ObservableState, <:Action}) = Float32.(observe(mdp.sim))

"""
Steps simulation and returns log probability of action.
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
Applies raw action to environment and returns reward.
"""
(mdp::ASTMDP{<:State, <:Action})(action) = (mdp)(convert_a(mdp, action))

"""
Applies converted action to environment and returns reward.
"""
function (mdp::ASTMDP{<:State, A})(a::A) where A <: Action
    # rewards (partial application)
    event = isevent(mdp.sim) ? mdp.reward.event_bonus : 0.0
    heur = mdp.reward.heuristic(distance(mdp.sim))
    rew = A <: SampleAction ? reward(mdp.sim, a.sample) : reward(mdp.sim)

    # stepping and likelihood evaluation
    logp = evaluate!(mdp, a)

    # final reward calculation
    r = mdp.reward.reward_function(logp, event, heur(mdp))
    r += rew isa Function ? rew(mdp.sim) : rew
    return r
end

"""
Returns Boolean indicating termination status.
"""
terminated(mdp::ASTMDP) = isterminal(mdp.sim) || isevent(mdp.sim)

# Connects AdaStress interface to CommonRLInterface

CommonRLInterface.reset!(mdp::ASTMDP) = reset!(mdp)

CommonRLInterface.actions(mdp::ASTMDP) = actions(mdp)

CommonRLInterface.observe(mdp::ASTMDP) = observe(mdp)

CommonRLInterface.act!(mdp::ASTMDP, action) = (mdp)(action)

CommonRLInterface.terminated(mdp::ASTMDP) = terminated(mdp)
