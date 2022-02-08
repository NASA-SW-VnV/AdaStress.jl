# ******************************************************************************************
# Notices:
#
# Copyright © 2022 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.
#
# Disclaimers
#
# No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF ANY KIND,
# EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY
# THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR FREEDOM FROM INFRINGEMENT, ANY
# WARRANTY THAT THE SUBJECT SOFTWARE WILL BE ERROR FREE, OR ANY WARRANTY THAT DOCUMENTATION,
# IF PROVIDED, WILL CONFORM TO THE SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT, IN ANY MANNER,
# CONSTITUTE AN ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS,
# RESULTING DESIGNS, HARDWARE, SOFTWARE PRODUCTS OR ANY OTHER APPLICATIONS RESULTING FROM
# USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY DISCLAIMS ALL WARRANTIES AND
# LIABILITIES REGARDING THIRD-PARTY SOFTWARE, IF PRESENT IN THE ORIGINAL SOFTWARE, AND
# DISTRIBUTES IT "AS IS."
#
# Waiver and Indemnity:  RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE UNITED
# STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT.
# IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS IN ANY LIABILITIES, DEMANDS, DAMAGES,
# EXPENSES OR LOSSES ARISING FROM SUCH USE, INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON,
# OR RESULTING FROM, RECIPIENT'S USE OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND
# HOLD HARMLESS THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL
# AS ANY PRIOR RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR ANY
# SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS AGREEMENT.
# ******************************************************************************************

"""
    MCTS <: LocalSolver

Monte Carlo tree search algorithm.
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
    MCTSResult <: LocalResult

Monte Carlo tree search result.
"""
mutable struct MCTSResult <: LocalResult
    path::Vector
end

"""
Roll out random policy from given state, returning total return and final state.
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
Perform one full simulation of MCTS.
"""
function simulate(mcts::MCTS, mdp::CommonRLInterface.AbstractEnv, s::Node=mcts.tree)
    s.n += 1

    # expansion
    s.n == 1 && return rollout(mdp, s)

    if length(s.transitions) < mcts.k * s.n ^ mcts.α
        # progressive widening
        a = rand(actions(mdp))
        s′ = add(s, a; forward=false)
    else
        # maximize upper confidence bound
        a, s′ = top_transition(s)
    end

    # transition
    d = terminated(mdp)
    r = act!(mdp, a)
    d && return r, s′

    # update value estimate
    rf, sf = simulate(mcts, mdp, s′)
    q = r + rf
    s′.q = ((s′.n - 1) * s′.q + q) / (s′.n)

    # update upper confidence bound
    s.transitions[a => s′] = s′.q + mcts.c * sqrt(log(s.n) / s′.n)
    return q, sf
end

function Solvers.solve(mcts::MCTS, env_fn::Function)
    mdp = env_fn()
    A = typeof(rand(actions(mdp)))
    mcts.tree = Node{A}()
    best_paths = PriorityQueue()

    @showprogress for _ in 1:mcts.num_iterations
        reset!(mdp)
        r, s = simulate(mcts, mdp)
        path = trace(s)
        enqueue!(best_paths, MCTSResult(path), r, mcts.top_k)
    end

    return best_paths
end
