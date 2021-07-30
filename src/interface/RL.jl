# Connects ASTInterface with CommonRLInterface, allowing compatibility with standard RL-based solvers.

function CommonRLInterface.reset!(mdp::ASTMDP)
    reset!(mdp.sim)
    reset!(mdp.heuristic, distance(mdp.sim))
    return nothing
end

CommonRLInterface.actions(mdp::ASTMDP) = environment(mdp.sim)

CommonRLInterface.observe(mdp::ASTMDP) = Float32.(observe(mdp.sim))

function CommonRLInterface.act!(mdp::ASTMDP, action::Vector{<:Real})
	env = environment(mdp.sim)
	value = unflatten(mdp, action)
	sample = create_sample(env, value, mdp.marginalize)

	step!(mdp.sim, value)
	r = reward(mdp, sample)
	return r
end

CommonRLInterface.terminated(mdp::ASTMDP) = isterminal(mdp.sim) || isevent(mdp.sim)
