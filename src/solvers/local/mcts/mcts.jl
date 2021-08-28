"""
Monte Carlo tree search object.
"""
Base.@kwdef mutable struct MCTS <: LocalSolver
    num_iterations::Int64      = 1000
    top_k::Int64               = 10
    k::Float64                 = 1.0
    alpha::Float64             = 0.5
    c::Float64                 = 1.0
    tree::Union{Node, Nothing} = nothing
end

"""
Selects maximum item from iterable collection as determined by supplied function.
"""
function Base.argmax(f::Function, itr)
    v = collect(itr)
    return v[argmax(f.(v))]
end

"""
Rolls out random policy from given state, returning total return and final state.
"""
function rollout(mdp::CommonRLInterface.AbstractEnv, s::Node)
    d = terminated(mdp)
    a = rand(actions(mdp))
    r = act!(mdp, a)
    if d
        return r, s
    else
        sp = add(s, a; forward=false)
        rf, sf = rollout(mdp, sp)
        return r + rf, sf
    end
end

"""
Performs one full simulation of MCTS.
"""
function simulate(mcts::MCTS, mdp::CommonRLInterface.AbstractEnv, s::Node=mcts.tree)
    s.n += 1

    # expansion
    if s.new
        s.new = false
        return rollout(mdp, s)
    end

    if length(children(s)) < mcts.k * s.n ^ mcts.alpha
        # progressive widening
        a = rand(actions(mdp))
        add(s, a)
    else
        # upper confidence bound
        ucb = a -> s.actions[a].q + mcts.c * sqrt(log(s.n) / s.actions[a].n)
        a = argmax(ucb, keys(s.actions))
    end

    # transition
    d = terminated(mdp)
    r = act!(mdp, a)

    if d
        return r, s
    else
        # update value estimate
        sp = s.actions[a]
        rf, sf = simulate(mcts, mdp, sp)
        q = r + rf
        sp.q = ((sp.n - 1) * sp.q + q) / (sp.n)
        return q, sf
    end
end

"""
Solve environment with MCTS.
"""
function Solvers.solve(mcts::MCTS, env_fn::Function)
    mdp = env_fn()
    A = typeof(rand(actions(mdp)))
    mcts.tree = Node{A}()
    best_paths = PriorityQueue()

    @showprogress for _ in 1:mcts.num_iterations
        reset!(mdp)
        r, s = simulate(mcts, mdp)
        path = trace_back(s)
        best_paths[deepcopy(path)] = r
        if length(best_paths) > mcts.top_k
            delete!(best_paths, first(keys(best_paths)))
        end
    end

    return best_paths
end
