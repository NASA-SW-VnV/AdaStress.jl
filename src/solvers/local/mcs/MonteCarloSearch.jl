module MonteCarloSearch

export
    MCS,
    solve

using ..Solvers
import ..Solvers.solve

using CommonRLInterface
using DataStructures: PriorityQueue
using ProgressMeter

Base.@kwdef mutable struct MCS <: LocalSolver
    num_steps::Int64=10000
    top_k::Int64=10
end

function Solvers.solve(mcs::MCS, env_fn::Function)
    mdp = env_fn()
    reset!(mdp)
    path = UInt32[]
    best_paths = PriorityQueue()
    ret = 0.0
    @showprogress for _ in 1:mcs.num_steps
        a = rand(actions(mdp))
        push!(path, a)
        ret += act!(mdp, a)
        if terminated(mdp)
            best_paths[deepcopy(path)] = ret
            if length(best_paths) > mcs.top_k
                delete!(best_paths, first(keys(best_paths)))
            end
            empty!(path)
            ret = 0.0
            reset!(mdp)
        end
    end

    return best_paths
end

end
