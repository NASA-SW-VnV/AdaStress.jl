"""
Solvers module. Provides algorithms for solving AST MDPs.
"""
module Solvers

using AdaStress: exclude, Interface
using DataStructures
import DataStructures: enqueue!

export
    solve,
    replay!,

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
