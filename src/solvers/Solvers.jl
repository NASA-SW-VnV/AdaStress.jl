"""
Provides solvers for AST MDPs.
"""
module Solvers

"""
Abstract base type for solvers.
"""
abstract type Solver end

"""
Abstract type for solvers that return an adversial policy.
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

end
