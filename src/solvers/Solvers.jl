"""
Provides solvers for AST MDPs.
"""
module Solvers

using DataStructures
import DataStructures: enqueue!

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
include("utils.jl")
include("global/sac/SoftActorCritic.jl")
include("local/mcs/MonteCarloSearch.jl")
include("local/mcts/MonteCarloTreeSearch.jl")

using .SoftActorCritic
using .MonteCarloSearch
using .MonteCarloTreeSearch

end
