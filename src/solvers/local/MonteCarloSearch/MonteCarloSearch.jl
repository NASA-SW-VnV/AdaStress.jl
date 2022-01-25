"""
Simple uninformed Monte Carlo search. Should be used as a baseline for other local solvers,
in terms of runtime performance and overall efficacy.
"""
module MonteCarloSearch

export
    MCS,
    solve

using ..Solvers
import ..Solvers.solve

using CommonRLInterface
using DataStructures: PriorityQueue, enqueue!
using ProgressMeter

"""
    MCS <: LocalSolver

Monte Carlo search solver.
"""
Base.@kwdef mutable struct MCS <: LocalSolver
    num_iterations::Int64 = 1000
    top_k::Int64          = 10
end

"""
    MCSResult <: LocalResult

Monte Carlo search result.
"""
mutable struct MCSResult <: LocalResult
    path::Vector
end

function Solvers.solve(mcs::MCS, env_fn::Function)
    mdp = env_fn()
    A = typeof(rand(actions(mdp)))
    best_paths = PriorityQueue()

    @showprogress for _ in 1:mcs.num_iterations
        reset!(mdp)
        path = A[]
        r = 0.0
        d = false
        while !d
            d = terminated(mdp)
            a = rand(actions(mdp))
            push!(path, a)
            r += act!(mdp, a)
            d && enqueue!(best_paths, MCSResult(path), r, mcs.top_k)
        end
    end

    return best_paths
end

end
