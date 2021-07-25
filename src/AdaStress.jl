"""
Adaptive stress testing (AST) tool. See documentation for usage.
"""
module AdaStress

__precompile__(false) #TODO: remove upon release

include("interface/ASTInterface.jl")
include("analysis/Analysis.jl")
include("solvers/Solvers.jl")

using .ASTInterface
using .Analysis
using .Solvers

export ASTInterface

end
