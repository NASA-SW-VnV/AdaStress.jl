"""
Implements the soft actor-critic algorithm with certain modifications enabling post-analysis.
"""
module SoftActorCritic

export
	SAC,
    solve,
    MLPActorCritic,
    Master2Worker,
    Worker2Master,
    simulation_task,
    distributed_solve

using ..Solvers

using BSON: @save
using CommonRLInterface
using CUDA
using Dates
using Distributed
using Distributions
using Flux
using Flux: params
using LinearAlgebra
using ProgressMeter
using Random
using Statistics
using Zygote

const DEFAULT_SAVE_DIR = joinpath(@__DIR__, "checkpoints")

include("core.jl")
include("replay_buffer.jl")
include("sac.jl")
include("distributed.jl")

end