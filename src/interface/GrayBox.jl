# To be implemented by user.

"""
Parent type for user simulation in which state and environment are exposed.
"""
abstract type GrayBox <: AbstractSimulation end


"""
Resets simulation.
"""
reset!(sim::GrayBox)::Nothing = unimplemented()


"""
Returns Environment object constructed in simulation.
Environment is an alias of Dict{Symbol, Sampleable}.
"""
environment(sim::GrayBox)::Environment = unimplemented()


"""
Returns quasi-normalized observation of simulation.
Observation should copy, not reference, simulation state.
If simulation is unobservable, leave unimplemented.
"""
observe(sim::GrayBox)::AbstractVector{<:Real} = unimplemented()


"""
Steps simulation given an EnvironmentValue object.
EnvironmentValue is an alias of Dict{Symbol, Any}.
"""
step!(sim::GrayBox, x::EnvironmentValue)::Nothing = unimplemented()


"""
Checks whether simulation has finished due to time limit or terminal state, independent of event status.
"""
isterminal(sim::GrayBox)::Bool = unimplemented()


"""
Checks whether simulation is in an event state.
"""
isevent(sim::GrayBox)::Bool = unimplemented()


"""
Returns custom metric of distance to event. For best results, metric should depend only on current state.
"""
distance(sim::GrayBox)::Real = unimplemented()


"""
Flattens environment variable to quasi-normalized array.
"""
flatten(distribution::Any, value::Any)::AbstractVector{<:Real} = unimplemented()
flatten(d::Distribution{Univariate, Continuous}, v::Real) = [(v - mean(d)) / std(d)]
flatten(d::Uniform, v::Real) = [(2 * v - d.a - d.b) / (d.b - d.a)]


"""
Reconstructs environment variable from quasi-normalized array.
"""
unflatten(distribution::Any, array::AbstractVector{<:Real})::Any = unimplemented()
unflatten(d::Distribution{Univariate, Continuous}, a::AbstractVector{<:Real}) = std(d) * a[] + mean(d)
unflatten(d::Uniform, a::AbstractVector{<:Real}) = (a[] * (d.b - d.a) + d.a + d.b) / 2

"""
Advanced option: additional reward or heuristic. Relies on partial function application to
allow efficient calculation of R(s,a,s') = R(s,a)(s'). Should not alter simulation. See
documentation for more details.
"""
reward(sim::GrayBox, x::EnvironmentValue) = 0.0

"""
Checks whether simulation encountered an event during episode. Should be implemented only
for episodic simulations where event data is available at end of episode.
"""
wasevent(sim::GrayBox)::Bool = unimplemented()

"""
Returns minimum distance to event across entire episode, also known as `miss distance`.
Should be implemented only for episodic simulations where distance data is available at end of episode.
"""
missdistance(sim::GrayBox)::Bool = unimplemented()
