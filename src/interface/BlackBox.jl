# To be implemented by user.

"""
Parent type for user simulation in which neither state nor environment is exposed.
"""
abstract type BlackBox <: AbstractSimulation end


"""
Resets simulation.
"""
reset!(sim::BlackBox)::Nothing = unimplemented()


"""
Returns quasi-normalized observation of simulation.
Observation should copy, not reference, simulation state.
If simulation is unobservable, leave unimplemented.
"""
observe(sim::BlackBox)::Vector{<:Real} = unimplemented()


"""
Steps simulation and returns log probability of environment.
If simulation uses global RNG, the one-argument step! should be implemented.
If simulation requires non-global RNG, the two-argument function should be implemented instead.
"""
step!(sim::BlackBox)::Real = unimplemented()
step!(rng::AbstractRNG, sim::BlackBox)::Real = step!(sim)


"""
Checks whether simulation has finished due to time limit or terminal state, independent of event status.
"""
isterminal(sim::BlackBox)::Bool = unimplemented()


"""
Checks whether simulation is in an event state.
"""
isevent(sim::BlackBox)::Bool = unimplemented()


"""
Returns custom metric of distance to event. For best results, metric should depend only on current state.
"""
distance(sim::BlackBox)::Real = unimplemented()

"""
Advanced option: additional reward or heuristic. Relies on partial function application to
allow efficient calculation of R(s,s') = R(s)(s'). Should not alter simulation. See
documentation for more details.
"""
reward(sim::BlackBox) = 0.0
