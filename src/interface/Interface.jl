"""
Defines interface between custom simulation and AST framework.
Supports various levels of observability and access.
"""
module Interface

using Bijections
using BSON
using CommonRLInterface
using Distributions
using Parameters
using Sockets
import Base: rand

export
    ASTMDP,
    BlackBox,
    GrayBox,
    Environment,
    EnvironmentValue,
    ASTClient,
    ASTServer,

    reset!,
    environment,
    observe,
    step!,
    isterminal,
    isevent,
    distance,
    flatten,
    unflatten,
    connect!,

    DistanceGradient,
    DistanceMinimum,
    DistanceNull,

    WeightedReward,
    VectorReward

include("heuristics.jl")
include("rewards.jl")
include("AST.jl")
include("BlackBox.jl")
include("GrayBox.jl")
include("RL.jl")
include("remote.jl")

end
