"""
Internal abstract type.
"""
abstract type AbstractSimulation end

"""
Internal unimplemented exception type.
"""
struct UnimplementedError <: Exception end
Base.showerror(io::IO, ::UnimplementedError) = print(io, "Required function has not been implemented.")
unimplemented() = throw(UnimplementedError())

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
Abstract type for AST state.
"""
abstract type State end

struct ObservableState <: State end

struct UnobservableState <: State end

"""
Abstract type for AST action.
"""
abstract type Action end

"""
Action corresponding to instantiation of stochastic environment.
"""
struct SampleAction <: Action
    sample::EnvironmentValue
end

"""
Action corresponding to setting random seed.
"""
struct SeedAction <: Action
    seed::UInt32
end

"""
Abstract AST MDP type.
"""
abstract type AbstractASTMDP{S<:State, A<:Action} <: CommonRLInterface.AbstractEnv end

"""
Standard MDP object for AST. Wraps simulation and contains auxiliary information and parameters.
#TODO: add hooks for annealing.
"""
Base.@kwdef mutable struct ASTMDP{S<:State, A<:Action} <: AbstractASTMDP{S, A}
	sim::AbstractSimulation						        # simulation wrapping system under test
    reward::Reward=Reward()                             # reward structure
	env_info::EnvironmentInfo=EnvironmentInfo()	        # inferred environment properties
    rng::AbstractRNG=Random.default_rng()               # RNG used for simulation
end
