#TODO: settle on solver-problem interaction
# 1) result = solve!(solver, problem)
# 2) result = solve!(solver(problem, args...))
# 3) result = (solver)(problem)

#TODO: should solvers accept MDP or MDP generating function
# - how is training/testing differentiated
# - are two separate environments really necessary?
# - need to consider multiprocessing versions as well

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
