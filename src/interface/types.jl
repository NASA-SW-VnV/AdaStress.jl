# ******************************************************************************************
# Notices:
#
# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.
#
# Disclaimers
#
# No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF ANY KIND,
# EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY
# THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR FREEDOM FROM INFRINGEMENT, ANY
# WARRANTY THAT THE SUBJECT SOFTWARE WILL BE ERROR FREE, OR ANY WARRANTY THAT DOCUMENTATION,
# IF PROVIDED, WILL CONFORM TO THE SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT, IN ANY MANNER,
# CONSTITUTE AN ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS,
# RESULTING DESIGNS, HARDWARE, SOFTWARE PRODUCTS OR ANY OTHER APPLICATIONS RESULTING FROM
# USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY DISCLAIMS ALL WARRANTIES AND
# LIABILITIES REGARDING THIRD-PARTY SOFTWARE, IF PRESENT IN THE ORIGINAL SOFTWARE, AND
# DISTRIBUTES IT "AS IS."
#
# Waiver and Indemnity:  RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE UNITED
# STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT.
# IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS IN ANY LIABILITIES, DEMANDS, DAMAGES,
# EXPENSES OR LOSSES ARISING FROM SUCH USE, INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON,
# OR RESULTING FROM, RECIPIENT'S USE OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND
# HOLD HARMLESS THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL
# AS ANY PRIOR RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR ANY
# SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS AGREEMENT.
# ******************************************************************************************

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

Base.zero(val::EnvironmentValue) = EnvironmentValue(k => zero(v) for (k, v) in val)

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
    num_steps::Int64          = 0                     # number of steps (zero if unknown)
	env_info::EnvironmentInfo = EnvironmentInfo()	  # inferred environment properties
    rng::AbstractRNG          = Random.default_rng()  # RNG used for simulation
end
