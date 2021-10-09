"""
Adaptive stress testing (AST) package. Finds and analyzes likeliest failures for a simulated
system under test. See documentation for usage.
"""
module AdaStress

include("utils.jl")
include("interface/Interface.jl")
include("solvers/Solvers.jl")
include("analysis/Analysis.jl")

using .Interface
using .Solvers
using .Analysis

export Interface

function __init__()
    init_submodules()
end

end
