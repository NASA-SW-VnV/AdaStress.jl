"""
Reward function abstract type.
Custom reward functions can inherit this type and implement application function.
For non-AST-specific reward functions, implement the `reward` function defined in th
"""
abstract type RewardFunction end

"""
Default reward function. Sums components at each timestep.
"""
Base.@kwdef mutable struct WeightedRewardFunction <: RewardFunction
    wl::Float64=1.0
    we::Float64=1.0
    wh::Float64=1.0
end

function (rf::WeightedRewardFunction)(logprob::Float64, event::Float64, heuristic::Float64)
    return rf.wl * logprob + rf.we * event + rf.wh * heuristic
end

"""
Vector reward function. Maintains separate components to facilitate post-analysis and
enhanced learning methods.
"""
struct VectorRewardFunction <: RewardFunction end

function (::VectorRewardFunction)(logprob::Float64, event::Float64, heuristic::Float64)
    return (logprob, event, heuristic)
end

"""
Standard AST reward structure.
"""
Base.@kwdef mutable struct Reward
    marginalize::Bool=true
    heuristic::DistanceHeuristic=GradientHeuristic()
    event_bonus::Float64=0.0
    reward_function::RewardFunction=WeightedRewardFunction()
end
