
"""
Abstract base type for solvers.
"""
abstract type Solver end

"""
Abstract type for solvers that return an adversarial policy.
"""
abstract type GlobalSolver <: Solver end

"""
Abstract type for solvers that return failure examples.
"""
abstract type LocalSolver <: Solver end

"""
Abstract type for solver output.
"""
abstract type Result end

"""
Abstract type for global result (adversarial policy).
"""
abstract type GlobalResult <: Result end

"""
Abstract type for local result (failure examples).
"""
abstract type LocalResult <: Result end

function (solver::Solver)(env::Any)
    @warn """Solver was not passed an environment generator.
    Suboptimal performance may occur if environment cannot be auto-replicated.
    """
    return solver(() -> env)
end

"""
    solve(::Solver, ::Function)

Apply solver to MDP generator function.
"""
function solve(::Solver, ::Function) end
(solver::Solver)(env_fn::Function) = solve(solver, env_fn)
