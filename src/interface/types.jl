"""
Internal abstract type.
"""
abstract type AbstractSimulation end

"""
Internal unimplemented exception type.
"""
Base.@kwdef struct UnimplementedError <: Exception
    msg::String = "Required function has not been implemented."
end
Base.showerror(io::IO, e::UnimplementedError) = print(io, e.msg)
unimplemented() = throw(UnimplementedError())

"""
Set of environment variables and their distributions.
"""
const Environment = Dict{Symbol, Sampleable}

"""
Samples from environment.
"""
const EnvironmentValue = Dict{Symbol, Any}

"""
Inferred properties of single environment variable.
"""
struct VariableInfo
	n::Int64	# environment variable dimensionality
	t::Type		# environment variable type
end

"""
Properties of environment variables.
"""
const EnvironmentInfo = Dict{Symbol, VariableInfo}

"""
Abstract type for AST state.
"""
abstract type State end

"""
State type for observable simulation.
"""
struct ObservableState <: State end

"""
State type for unobservable simulation.
"""
struct UnobservableState <: State end

"""
Abstract type for AST action.
"""
abstract type Action end

"""
Action type corresponding to instantiation of stochastic environment.
"""
struct SampleAction <: Action
    sample::EnvironmentValue
end

"""
Action type corresponding to setting random seed of simulation.
"""
struct SeedAction <: Action
    seed::UInt32
end

"""
Abstract AST MDP type.
"""
abstract type AbstractASTMDP{S<:State, A<:Action} <: CommonRLInterface.AbstractEnv end

"""
Return state type.
"""
state_type(::AbstractASTMDP{S, A}) where {S, A} = S

"""
Return action type.
"""
action_type(::AbstractASTMDP{S, A}) where {S, A} = A

"""
Abstract AST reward structure.
"""
abstract type AbstractReward end

"""
Standard MDP object for AST. Wraps simulation and contains auxiliary information and parameters.
#TODO: add hooks for annealing.
"""
Base.@kwdef mutable struct ASTMDP{S<:State, A<:Action} <: AbstractASTMDP{S, A}
	sim::AbstractSimulation						      # simulation wrapping system under test
    reward::AbstractReward    = Reward()              # reward structure
    episodic::Bool            = false                 # episodic evaluation
	env_info::EnvironmentInfo = EnvironmentInfo()	  # inferred environment properties
    rng::AbstractRNG          = Random.default_rng()  # RNG used for simulation
end
