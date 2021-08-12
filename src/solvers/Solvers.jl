"""
Provides solvers for AST MDPs.
"""
module Solvers

using CommonRLInterface

export
    GlobalSolver,
    LocalSolver,
    GlobalResult,
    LocalResult,

    SoftActorCritic

include("types.jl")
include("global/sac/SoftActorCritic.jl")

using .SoftActorCritic

end
