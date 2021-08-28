"""
Provides solvers for AST MDPs.
"""
module Solvers

export
    solve,

    GlobalSolver,
    LocalSolver,
    GlobalResult,
    LocalResult,

    SoftActorCritic,
    MonteCarloSearch,
    MonteCarloTreeSearch

include("types.jl")
include("global/sac/SoftActorCritic.jl")
include("local/mcs/MonteCarloSearch.jl")
include("local/mcts/MonteCarloTreeSearch.jl")

using .SoftActorCritic
using .MonteCarloSearch
using .MonteCarloTreeSearch

end
