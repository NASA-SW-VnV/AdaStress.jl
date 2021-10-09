# To be implemented by user.

"""
    BlackBox <: AbstractSimulation

Parent type for user simulation in which environment is not exposed.
"""
abstract type BlackBox <: AbstractSimulation end

"""
    reset!(sim::BlackBox)

Reset simulation.
"""
reset!(sim::BlackBox) = unimplemented()

"""
    observe(sim::BlackBox)

Return quasi-normalized observation of simulation. Observation should copy, not reference,
simulation state. If simulation is unobservable, leave unimplemented.
"""
observe(sim::BlackBox)::AbstractVector{<:Real} = unimplemented()

"""
    step([rng::AbstractRNG], sim::BlackBox)

Step simulation and return log probability of transition. If simulation uses global RNG,
the one-argument step! should be implemented. If simulation requires non-global RNG, the
two-argument function should be implemented instead.
"""
step!(sim::BlackBox)::Real = unimplemented()
step!(rng::AbstractRNG, sim::BlackBox)::Real = step!(sim)

"""
    isterminal(sim::BlackBox)

Check whether simulation has finished due to time limit or terminal state, independent of
failure status.
"""
isterminal(sim::BlackBox)::Bool = unimplemented()

"""
   isevent(sim::BlackBox)

Check whether simulation is in event (failure) state. For episodic simulation, check whether
event occurred during episode.
"""
isevent(sim::BlackBox)::Bool = unimplemented()

"""
    distance(sim::BlackBox)

Return custom metric of distance to event. For best results, metric should depend only on
current state. For episodic simulation, return minimum distance to event across entire
episode (miss distance).
"""
distance(sim::BlackBox)::Real = unimplemented()

"""
    reward(sim::BlackBox)

Advanced option: additional reward or heuristic. Returns scalar value or partially-applied
function, allowing efficient calculation of R(s,s') = R(s)(s'). Should not alter simulation.
See documentation for more details.
"""
reward(sim::BlackBox) = 0.0
