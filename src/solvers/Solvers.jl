module Solvers

abstract type Solver end

"""
Returns adversial policy.
"""
abstract type GlobalSolver <: Solver end

"""
Returns failure examples.
"""
abstract type LocalSolver <: Solver end

end
