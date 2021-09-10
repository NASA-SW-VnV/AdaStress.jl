"""
AST core objective function abstract type. Defines fundamental reward function by combining
only log probability, event bonus, and distance heuristic. Additional contributions to the
reward should be added by implementing `reward` function in GrayBox or BlackBox interface.
"""
abstract type AbstractCoreObjective end

"""
Default objective function. Sums components at each timestep.
"""
Base.@kwdef mutable struct WeightedObjective <: AbstractCoreObjective
    wl::Float64 = 1.0
    we::Float64 = 1.0
    wh::Float64 = 1.0
end

function (rf::WeightedObjective)(logprob::Float64, event::Float64, heuristic::Float64)
    return rf.wl * logprob + rf.we * event + rf.wh * heuristic
end

"""
Vector reward function. Maintains separate components to facilitate post-analysis and
enhanced learning methods.
"""
struct VectorObjective <: AbstractCoreObjective end

function (::VectorObjective)(logprob::Float64, event::Float64, heuristic::Float64)
    return (logprob, event, heuristic)
end

"""
Standard AST reward structure.
"""
abstract type AbstractReward end

Base.@kwdef mutable struct Reward <: AbstractReward
    marginalize::Bool                      = true
    heuristic::AbstractDistanceHeuristic   = GradientHeuristic()
    event_bonus::Float64                   = 0.0
    reward_function::AbstractCoreObjective = WeightedObjective()
end
