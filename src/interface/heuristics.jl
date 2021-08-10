"""
Distance heuristic abstract type.
Custom heuristics can inherit this type and implement `reset!` and application functions.
"""
abstract type DistanceHeuristic end

"""
Gradient of conservative potential. Default and recommended.
"""
@with_kw mutable struct DistanceGradient <: DistanceHeuristic
    d_curr::Float64=NaN
end

function reset!(dh::DistanceGradient, d::Float64)
    dh.d_curr = d
end

function (dh::DistanceGradient)(d::Float64)
    Δd = d - dh.d_curr
    dh.d_curr = d
    return -Δd
end

"""
Minimum distance across episode. Warning: non-Markovian.
"""
@with_kw mutable struct DistanceMinimum <: DistanceHeuristic
    d_min::Float64=NaN
end

function reset!(dh::DistanceMinimum, d::Float64)
    dh.d_min = d
end

function (dh::DistanceMinimum)(d::Float64)
    dh.d_min = min(dh.d_min, d)
    return -dh.d_min
end

"""
Null heuristic. Returns zero.
"""
struct DistanceNull <: DistanceHeuristic end
reset!(::DistanceNull, ::Float64) = nothing
(::DistanceNull)(::Float64) = 0.0
