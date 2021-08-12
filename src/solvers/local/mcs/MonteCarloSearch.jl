module MonteCarloSearch

using CommonRLInterface
using ProgressMeter

export
    MCS,
    solve

Base.@kwdef mutable struct MCS
    num_steps::Int64=10000
end

function solve(mcs::MCS, mdp::CommonRLInterface.AbstractEnv)
    reset!(mdp)
    path, path_best = [], []
    ret, ret_best = 0.0, -Inf

    @showprogress for _ in 1:mcs.num_steps
        a = rand(actions(mdp))
        push!(path, a)
        ret += act!(mdp, a)
        if terminated(mdp)
            if ret > ret_best
                path_best = deepcopy(path)
                ret_best = ret
            end
            empty!(path)
            ret = 0.0
            reset!(mdp)
        end
    end

    return path_best
end

end
