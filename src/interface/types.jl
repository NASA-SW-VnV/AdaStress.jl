"""
Internal abstract type.
"""
abstract type AbstractSimulation end

"""
Stores distributions of environment variables.
"""
const Environment = Dict{Symbol, Sampleable}

"""
Stores values of environment samples.
"""
const EnvironmentValue = Dict{Symbol, Any}

"""
Stores inferred properties of environment variable.
"""
struct VariableInfo
	n::Int64	# environment variable dimensionality
	t::Type		# environment variable type
end

"""
Stores properties of environment variables.
"""
const EnvironmentInfo = Dict{Symbol, VariableInfo}

"""
Abstract type for AST action.
"""
abstract type Action end

"""
Action corresponding to instantiation of stochastic environment.
#TODO: use internal field?
"""
struct SampleAction <: Action
    sample::EnvironmentValue
end

"""
Action corresponding to setting random seed.
#TODO: use internal field?
"""
struct SeedAction <: Action
    seed::UInt32
end

"""
MDP object for AST. Wraps simulation and contains auxiliary information and parameters.
#TODO: add hooks for annealing.
"""
Base.@kwdef mutable struct ASTMDP{A<:Action} <: CommonRLInterface.AbstractEnv
	sim::AbstractSimulation						        # simulation wrapping system under test
	reward_bonus::Float64=0.0   				        # bonus for reaching event
    reward::RewardFunction=WeightedReward()             # reward function
    marginalize::Bool=true                              # use marginalized probabilities
    heuristic::DistanceHeuristic=DistanceGradient()     # distance heuristic
	env_info::EnvironmentInfo=EnvironmentInfo()	        # inferred environment properties
end
