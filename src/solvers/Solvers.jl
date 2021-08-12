"""
Provides solvers for AST MDPs.
"""
module Solvers

export
    GlobalSolver,
    LocalSolver,
    GlobalResult,
    LocalResult,

    SoftActorCritic,
    MonteCarloSearch

include("types.jl")
include("global/sac/SoftActorCritic.jl")
include("local/mcs/MonteCarloSearch.jl")

using .SoftActorCritic
using .MonteCarloSearch

end
