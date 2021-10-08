"""
Solvers module. Provides algorithms for solving AST MDPs.
"""
module Solvers

using AdaStress: exclude
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
exclude("global/SoftActorCritic")
include("local/MonteCarloSearch/MonteCarloSearch.jl")
include("local/MonteCarloTreeSearch/MonteCarloTreeSearch.jl")

using .MonteCarloSearch
using .MonteCarloTreeSearch

end
