"""
Adaptive stress testing (AST) tool. See documentation for usage.
"""
module AdaStress

__precompile__(false) #TODO: remove upon release

include("interface/Interface.jl")
include("analysis/Analysis.jl")
include("solvers/Solvers.jl")

using .Interface
using .Analysis
using .Solvers

export Interface


end
