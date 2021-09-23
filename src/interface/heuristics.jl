"""
Distance heuristic abstract type.
Custom heuristics can inherit this type and implement `reset!` and application functions.
"""
abstract type AbstractDistanceHeuristic end
reset!(::AbstractDistanceHeuristic) = nothing
(::AbstractDistanceHeuristic)(::ASTMDP) = unimplemented()

"""
Gradient of conservative potential. Default and recommended.
"""
struct GradientHeuristic <: AbstractDistanceHeuristic end

function (::GradientHeuristic)(mdp::ASTMDP)
    d = distance(mdp.sim)
    @defer d - distance(mdp′.sim)
end

"""
Minimum distance across episode. Warning: non-Markovian.
"""
Base.@kwdef mutable struct MinimumHeuristic <: AbstractDistanceHeuristic
    d_min::Float64 = Inf
end

function (h::MinimumHeuristic)(mdp::ASTMDP)
    h.d_min = min(h.d_min, distance(mdp.sim))
    done = terminated(mdp)
    @defer done ? -h.d_min : 0.0
end

function reset!(h::MinimumHeuristic)
    h.d_min = Inf
end

"""
Retroactive minimum distance across episodic, for episodic MDPs.
"""
struct MissHeuristic <: AbstractDistanceHeuristic end

function (::MissHeuristic)(mdp::ASTMDP)
    d_miss = isterminal(mdp.sim) ? missdistance(mdp.sim) : 0.0
    @defer d_miss
end

"""
Null heuristic. Returns zero.
"""
struct NullHeuristic <: AbstractDistanceHeuristic end
(::NullHeuristic)(::ASTMDP) = @defer 0.0
