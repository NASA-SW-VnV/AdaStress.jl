"""
Simulation of airspace consisting of `n` aircraft equipped with collision avoidance systems.
"""
module SimpleACAS

using AdaStress
using Dates
using Distributions
using IterTools
using LinearAlgebra
using Plots
using ProgressMeter
using Random

export visualize

include("utils.jl")
include("aircraft.jl")
include("encounter.jl")
include("cas.jl")
include("response.jl")
include("metrics.jl")
include("simulator.jl")
include("logging.jl")
include("visualization.jl")

end
