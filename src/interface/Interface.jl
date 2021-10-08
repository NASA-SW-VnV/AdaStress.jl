"""
Interface between custom simulation and AST framework. Supports various levels of
observability and access.
"""
module Interface

using Bijections
using BSON
using CommonRLInterface
using Distributions
using Random
using Sockets
import Base: rand

export ASTMDP, BlackBox, GrayBox, Environment, EnviornmentValue
export VariableInfo, ObservableState, UnobservableState, SampleAction, SeedAction
export reset!, environment, observe, step!, isterminal, isevent, distance, flatten, unflatten
export Reward, WeightedObjective, VectorObjective
export GradientHeuristic, MinimumHeuristic, FinalHeuristic, NullHeuristic
export ASTClient, ASTServer, RemoteASTMDP, connect!, disconnect!, ping

include("types.jl")
include("utils.jl")
include("heuristics.jl")
include("BlackBox.jl")
include("GrayBox.jl")
include("rewards.jl")
include("AST.jl")
include("RL.jl")
include("remote/client.jl")
include("remote/server.jl")
include("remote/tunnel.jl")

end
