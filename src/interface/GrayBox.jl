# To be implemented by user.

"""
Parent type for user simulation in which state and environment are exposed.
"""
abstract type GrayBox <: AbstractSimulation end


"""
Resets simulation.
"""
function reset!(sim::GrayBox)::Nothing end


"""
Returns Environment object constructed in simulation.
Environment is an alias of Dict{Symbol, Sampleable}.
"""
function environment(sim::GrayBox)::Environment end


"""
Returns quasi-normalized observation of simulation.
Observation should copy, not reference, simulation state.
"""
function observe(sim::GrayBox)::Vector{<:Real} end


"""
Steps simulation given an EnvironmentValue object.
EnvironmentValue is an alias of Dict{Symbol, Any}.
"""
function step!(sim::GrayBox, x::EnvironmentValue)::Nothing end


"""
Checks whether simulation has finished due to time limit or terminal state, independent of event status.
"""
function isterminal(sim::GrayBox)::Bool end


"""
Checks whether simulation is in an event state.
"""
function isevent(sim::GrayBox)::Bool end


"""
Returns custom metric of distance to event. For best results, metric should depend only on current state.
"""
function distance(sim::GrayBox)::Real end


"""
Flattens environment variable to quasi-normalized array.
"""
function flatten(distribution::Any, value::Any)::Vector{<:Real} end

flatten(d::Distribution{Univariate, Continuous}, v::Real) = [(v - mean(d)) / std(d)]
flatten(d::Uniform, v::Real) = [(2 * v - d.a - d.b) / (d.b - d.a)]


"""
Reconstructs environment variable from quasi-normalized array.
"""
function unflatten(distribution::Any, array::Vector{<:Real})::Any end

unflatten(d::Distribution{Univariate, Continuous}, a::Vector{<:Real}) = std(d) * a[] + mean(d)
unflatten(d::Uniform, a::Vector{<:Real}) = (a[] * (d.b - d.a) + d.a + d.b) / 2
