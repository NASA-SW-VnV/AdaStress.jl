"""
Monte Carlo tree search object.
"""
Base.@kwdef mutable struct MCTS <: LocalSolver
    num_iterations::Int64      = 1000
    top_k::Int64               = 10
    k::Float64                 = 1.0
    α::Float64                 = 0.7
    c::Float64                 = 1.0
    tree::Union{Node, Nothing} = nothing
end

"""
Rolls out random policy from given state, returning total return and final state.
"""
function rollout(mdp::CommonRLInterface.AbstractEnv, s::Node)
    d = terminated(mdp)
    a = rand(actions(mdp))
    r = act!(mdp, a)
    d && return r, s

    s′ = add(s, a; forward=false)
    rf, sf = rollout(mdp, s′)
    return r + rf, sf
end

"""
Performs one full simulation of MCTS.
"""
function simulate(mcts::MCTS, mdp::CommonRLInterface.AbstractEnv, s::Node=mcts.tree)
    s.n += 1

    # expansion
    s.n == 1 && return rollout(mdp, s)

    # progressive widening or maximizing upper confidence bound
    pw = length(s.transitions) < mcts.k * s.n ^ mcts.α
    a, s′ = pw ? (rand(actions(mdp)), nothing) : top_transition(s)

    # transition
    d = terminated(mdp)
    r = act!(mdp, a)
    d && return r, s

    # update value estimate
    s′ = pw ? add(s, a) : s′
    rf, sf = simulate(mcts, mdp, s′)
    q = r + rf
    s′.q = ((s′.n - 1) * s′.q + q) / (s′.n)

    # update upper confidence bound
    s.transitions[a => s′] = s′.q + mcts.c * sqrt(log(s.n) / s′.n)
    return q, sf
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
        path = trace(s)
        enqueue!(best_paths, path, r, mcts.top_k)
    end

    return best_paths
end
