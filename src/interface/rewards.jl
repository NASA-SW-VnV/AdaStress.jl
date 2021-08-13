"""
Reward function abstract type.
Custom reward functions can inherit this type and implement application function.
"""
abstract type RewardFunction end

"""
Default reward function. Sums components at each timestep.
"""
Base.@kwdef mutable struct WeightedReward <: RewardFunction
    w_logprob::Float64=1.0
    w_event::Float64=1.0
    w_heuristic::Float64=1.0
end

function (r::WeightedReward)(logprob::Float64, event::Bool, heuristic::Float64, bonus::Float64)
    return r.w_logprob * logprob + r.w_event * (event ? bonus : 0.0) + r.w_heuristic * heuristic
end

"""
Vector reward function. Maintains separate components to facilitate post-analysis and
enhanced learning methods.
"""
struct VectorReward <: RewardFunction end

function (::VectorReward)(logprob::Float64, event::Bool, heuristic::Float64, bonus::Float64)
    return (logprob, event ? bonus : 0.0, heuristic)
end
