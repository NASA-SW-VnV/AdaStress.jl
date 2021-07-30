"""
Provides solvers for AST MDPs.
"""
module Solvers

include("types.jl")

export
    GlobalSolver,
    LocalSolver,
    GlobalResult,
    LocalResult,

    SoftActorCritic

include("global/sac/SoftActorCritic.jl")

using .SoftActorCritic

end
