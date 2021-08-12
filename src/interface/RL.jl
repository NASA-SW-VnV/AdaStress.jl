# Connects ASTInterface with CommonRLInterface, allowing compatibility with standard RL-based solvers.

function CommonRLInterface.reset!(mdp::ASTMDP)
    reset!(mdp.sim)
    reset!(mdp.heuristic, distance(mdp.sim))
    return nothing
end

CommonRLInterface.actions(mdp::ASTMDP{SampleAction}) = environment(mdp.sim)

CommonRLInterface.actions(::ASTMDP{SeedAction}) = UInt32

CommonRLInterface.observe(mdp::ASTMDP) = Float32.(observe(mdp.sim))

function CommonRLInterface.act!(mdp::ASTMDP{SampleAction}, action::Vector{<:Real})
    env = environment(mdp.sim)
	value = unflatten(mdp, action)
    logp = logprob(env, value, mdp.marginalize)
    step!(mdp.sim, value)

    event = isevent(mdp.sim)
    heuristic = mdp.heuristic(distance(mdp.sim))
	return mdp.reward(logp, event, heuristic, mdp.reward_bonus)
end

function CommonRLInterface.act!(mdp::ASTMDP{SeedAction}, seed::UInt32)
    logp = @fix seed step!(mdp.sim)
    event = isevent(mdp.sim)
    heuristic = mdp.heuristic(distance(mdp.sim))
	return mdp.reward(logp, event, heuristic, mdp.reward_bonus)
end

CommonRLInterface.terminated(mdp::ASTMDP) = isterminal(mdp.sim) || isevent(mdp.sim)
