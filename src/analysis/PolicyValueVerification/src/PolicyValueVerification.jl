"""
Analyze AST results containing a policy and value function represented by ensembles of
neural networks.
"""
module PolicyValueVerification

export
    CrossSection,
    BinaryRefinery,
    IntervalRefinery,

    cross_section,
    get_root,
    mean_network,
    spread_network,
    print_metrics,
    visualize,
    visualize!,
    analyze

using AdaStress: GlobalResult

using BSON
using Distributed
using Flux
using LazySets: HalfSpace
using LinearAlgebra
using NeuralVerification
using NeuralVerification: Hyperrectangle, Layer, compute_output
using Plots
using ProgressMeter
using RegionTrees
using RegionTrees: AbstractRefinery
using SharedArrays
using StaticArrays

BLAS.set_num_threads(1)

include("network.jl")
include("verification.jl")
include("multiprocessing.jl")
include("analysis.jl")
include("visualization.jl")

function __init__()
    @eval NeuralVerification σ = LazySets.σ # fixes Requires bug in NeuralVerification
end

end
