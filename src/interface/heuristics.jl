"""
Distance heuristic abstract type.
Custom heuristics can inherit this type and implement `reset!` and `apply!` functions.
Application can return value or partially-applied function transforming f(s,s') to f(s)(s').
"""
abstract type AbstractDistanceHeuristic end
reset!(::AbstractDistanceHeuristic) = nothing
apply!(::AbstractDistanceHeuristic, ::ASTMDP) = unimplemented()
(adh::AbstractDistanceHeuristic)(mdp::ASTMDP) = Functoid(apply!(adh, mdp))

"""
Gradient of conservative potential. Default and recommended.
"""
struct GradientHeuristic <: AbstractDistanceHeuristic end

function apply!(::GradientHeuristic, mdp::ASTMDP)
    d = distance(mdp.sim)
    return mdp′::ASTMDP -> d - distance(mdp′.sim)
end

"""
Minimum distance across episode. Warning: non-Markovian.
"""
Base.@kwdef mutable struct MinimumHeuristic <: AbstractDistanceHeuristic
    d_min::Float64 = Inf
end

function apply!(h::MinimumHeuristic, mdp::ASTMDP)
    h.d_min = min(h.d_min, distance(mdp.sim))
    return terminated(mdp) ? -h.d_min : 0.0
end

function reset!(h::MinimumHeuristic)
    h.d_min = Inf
end

"""
Final distance encountered. If MDP is episodic, this is equal to the minimum distance
(determined retroactively instead of accumulated as in MinimumHeuristic).
"""
struct FinalHeuristic <: AbstractDistanceHeuristic end
apply!(::FinalHeuristic, mdp::ASTMDP) = isterminal(mdp.sim) ? -distance(mdp.sim) : 0.0

"""
Null heuristic. Returns zero.
"""
struct NullHeuristic <: AbstractDistanceHeuristic end
apply!(::NullHeuristic, ::ASTMDP) = 0.0
