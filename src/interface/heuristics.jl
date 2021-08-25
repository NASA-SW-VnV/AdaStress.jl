"""
Distance heuristic abstract type.
Custom heuristics can inherit this type and implement `reset!` and application functions.
"""
abstract type DistanceHeuristic end
reset!(::DistanceHeuristic) = nothing

"""
Gradient of conservative potential. Default and recommended.
"""
struct GradientHeuristic <: DistanceHeuristic end
(::GradientHeuristic)(d::Float64) = @defer d - distance(mdp.sim)

"""
Minimum distance across episode. Warning: non-Markovian.
"""
Base.@kwdef struct MinimumHeuristic <: DistanceHeuristic
    d_min::Float64=Inf
end

function (h::MinimumHeuristic)(d::Float64)
    h.d_min = min(h.d_min, d)
    @defer -h.d_min
end

function reset!(h::MinimumHeuristic)
    h.d_min = Inf
end

"""
Null heuristic. Returns zero.
"""
struct NullHeuristic <: DistanceHeuristic end
(::NullHeuristic)(::Float64) = @defer 0.0
