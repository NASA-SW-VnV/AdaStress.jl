module Interface

__precompile__(false)

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

    reset!,
    environment,
    observe,
    step!,
    isterminal,
    isevent,
    distance,
    flatten,
    unflatten,

    DistanceGradient,
    DistanceMinimum,
    DistanceNull,
    DistanceCustom

include("heuristics.jl")
include("AST.jl")
include("BlackBox.jl")
include("GrayBox.jl")
include("RL.jl")
include("remote.jl")

end
