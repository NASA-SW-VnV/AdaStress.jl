"""
Monte Carlo tree search algorithm. Balances exploration and experience. Constructs
partially-ordered search tree over actions, treating states as anonymous.
"""
module MonteCarloTreeSearch

export
    MCTS,
    solve

using ..Solvers
import ..Solvers.solve

using CommonRLInterface
using DataStructures: PriorityQueue, enqueue!
using ProgressMeter

include("tree.jl")
include("mcts.jl")

end
