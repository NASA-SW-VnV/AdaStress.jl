# ******************************************************************************************
# Notices:
#
# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.
#
# Disclaimers
#
# No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF ANY KIND,
# EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY
# THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR FREEDOM FROM INFRINGEMENT, ANY
# WARRANTY THAT THE SUBJECT SOFTWARE WILL BE ERROR FREE, OR ANY WARRANTY THAT DOCUMENTATION,
# IF PROVIDED, WILL CONFORM TO THE SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT, IN ANY MANNER,
# CONSTITUTE AN ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS,
# RESULTING DESIGNS, HARDWARE, SOFTWARE PRODUCTS OR ANY OTHER APPLICATIONS RESULTING FROM
# USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY DISCLAIMS ALL WARRANTIES AND
# LIABILITIES REGARDING THIRD-PARTY SOFTWARE, IF PRESENT IN THE ORIGINAL SOFTWARE, AND
# DISTRIBUTES IT "AS IS."
#
# Waiver and Indemnity:  RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE UNITED
# STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT.
# IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS IN ANY LIABILITIES, DEMANDS, DAMAGES,
# EXPENSES OR LOSSES ARISING FROM SUCH USE, INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON,
# OR RESULTING FROM, RECIPIENT'S USE OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND
# HOLD HARMLESS THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL
# AS ANY PRIOR RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR ANY
# SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS AGREEMENT.
# ******************************************************************************************

TANH_APPROX = nothing

"""
Load tanh approximator from bson if uninitiated.
"""
function load_tanh()
    if TANH_APPROX === nothing
        global TANH_APPROX = BSON.load(joinpath(@__DIR__, "tanh.bson"), @__MODULE__)[:network]
    end
    return TANH_APPROX
end

"""
Retrain neural network univariate tanh approximator. Ensures f(x) = f(-x) and |f(x)| <= 1.
"""
function retrain_tanh(; depth::Int64=1, width::Int64=10, num_iter::Int64=10000, num_samples::Int64=1000)
    # Create model
    layers = Dense[]
    push!(layers, Dense(1, width, relu))
    for _ in 1:(depth - 1)
        push!(layers, Dense(width, width, relu))
    end
    push!(layers, Dense(width, 1))
    model = Chain(layers...)
    ps = Flux.params(model)
    opt = AdaBelief(1e-3)

    # Train network
    p = Progress(num_iter)
    loss = Ref(0.0)
    for _ in 1:num_iter
        x = randn(1, num_samples)
        gs = gradient(ps) do
            loss[] = sum((tanh.(x) .- model(x)).^2)
        end
        Flux.update!(opt, ps, gs)
        ProgressMeter.next!(p; showvalues = [(:loss, loss[])])
    end

    # Antisymmetrize
    d1 = Dense(reshape([1.0f0, -1.0f0], 2, 1))
    d2 = Dense(reshape([0.5f0, -0.5f0], 1, 2))
    layers = Dense[]
    for d in m.layers
        W = [d.W zero(d.W); zero(d.W) d.W]
        b = [d.b; d.b]
        push!(layers, Dense(W, b, d.σ))
    end

    # Clamp
    u, v = -1.0f0, 1.0f0
    d3 = Dense(reshape([1.0f0, 1.0f0], 2, 1), [-u, -v], relu)
    d4 = Dense(reshape([1.0f0, -1.0f0], 1, 2), [u])

    global TANH_APPROX = Chain(d1, layers..., d2, d3, d4)
end

"""
Extended neural network containing metadata about transformations.
"""
Base.@kwdef mutable struct ExtendedNetwork
    nnet::Network
    step_up::Int64      # index of step-up layer
    step_down::Int64    # index of step-down layer
    scale::Int64 = 0.0  # distance scale (for reward shaping)
end

"""
Convert Flux activation functions to nnet functions. Limited to `identity` and `relu`.
"""
function convert_f(f::Function)
    fsym = Symbol(f)
    if fsym == :identity
        return NeuralVerification.Id()
    elseif fsym == :relu
        return NeuralVerification.ReLU()
    else
        error("Unable to convert $f.")
    end
end

"""
Pad front of matrix with zeros.
"""
function fpad(M::Matrix{T}, dims::Tuple{Int64,Int64}) where T
    i, j = size(M)
    ip, jp = dims
    Mp = hcat(zeros(T, i + ip, jp), vcat(zeros(T, ip, j), M))
    return Mp
end

"""
Pad front of vector with zeros.
"""
function fpad(v::Vector{T}, sz::Int64) where T
    return vcat(zeros(T, sz), v)
end

"""
Pad front of matrix with identity.
"""
function fpad_id(M::Matrix{T}, sz::Int64) where T
    Mp = fpad(M, (sz, sz))
    Mp[1:sz,1:sz] = Matrix{T}(I, sz, sz)
    return Mp
end

"""
Pad front of Flux.Dense layer and convert to nnet layer.
"""
function fpad(l::Dense, sz::Int64)
    W = fpad_id(l.weight, sz)
    b = fpad(l.bias, sz)
    σ = convert_f(l.σ)
    return Layer(W, b, σ)
end

"""
Construct block diagonal matrix from matrix arguments.
"""
function block_diag(matrices...)
    m = sum((M -> size(M, 1)).(matrices))
    n = sum((M -> size(M, 2)).(matrices))
    B = zeros(Float32, m, n)

    i, j = 1, 1
    for M in matrices
        mi, mj = size(M)
        B[i:i+mi-1, j:j+mj-1] = M
        i += mi
        j += mj
    end
    return B
end

"""
Construct block diagonal matrix repetition.
"""
function rep_diag(M::Matrix, n::Int64)
    A = [[i == j ? M : zero(M) for i in 1:n] for j in 1:n]
    return reduce(hcat, reduce(vcat, m) for m in A)
end

"""
Construct network that squashes input elementwise with tanh approximator, provided lower
and upper bounds.
"""
function squash_model(ls::Vector{Float64}, us::Vector{Float64})
    # Stack of tanh approximators
    n = length(ls)
    layers = Dense[]
    for d in load_tanh().layers
        W = rep_diag(d.W, n)
        b = repeat(d.b, n)
        push!(layers, Dense(W, b, d.σ))
    end

    # Rescale
    ls, us = Float32.(ls), Float32.(us)
    W = diagm((us - ls) / 2)
    b = (us + ls) / 2
    push!(layers, Dense(W, b))

    return Chain(layers...)
end

"""
Construct layer that computes mean of previous layer.
"""
function mean_layer(sz::Int64)
    W = Float32.(ones(1, sz) / sz)
    b = zeros(Float32, 1)
    σ = NeuralVerification.Id()
    return Layer(W, b, σ)
end

"""
Construct layer that computes elementwise absolute value of previous layer.
"""
function abs_layers(sz::Int64)
    M = Matrix{Float32}(I, sz, sz)
    W1 = vcat(M, -M)
    b1 = zeros(Float32, 2 * sz)
    σ1 = NeuralVerification.ReLU()

    W2 = hcat(M, M)
    b2 = zeros(Float32, sz)
    σ2 = NeuralVerification.Id()

    return (Layer(W1, b1, σ1), Layer(W2, b2, σ2))
end

"""
Coalesce artificially constructed network, incorporating layers without activation functions
into subsequent layers. Note that this is not guaranteed to reduce the number of parameters,
only the number of layers.
"""
function coalesce(nnet::Network)
    layers = []
    W, b, stored = nothing, nothing, false
    for l in nnet.layers
        W = stored ? l.weights * W : l.weights
        b = stored ? l.weights * b + l.bias : l.bias
        stored = l.activation isa NeuralVerification.Id && l != nnet.layers[end]
        stored || push!(layers, Layer(W, b, l.activation))
    end
    return Network(layers)
end

"""
Convert policy from actor-critic object to neural network. Concatenation and squashing are
represented by ReLU configurations.
"""
function policy_network(ac::GlobalResult; act_mins::Vector{Float64}, act_maxs::Vector{Float64})
    # Dimensions and parameters
    n_obs = size(ac.pi.net.layers[1].weight, 2)
    n_act = size(ac.pi.mu_layer.weight, 1)
    layers = []

    # Add duplication layer
    M = Matrix{Float32}(I, n_obs, n_obs)
    W = vcat(M, M)
    b = zeros(Float32, 2*n_obs)
    σ = NeuralVerification.Id()
    push!(layers, Layer(W, b, σ))
    step_up = length(layers)

    # Add policy net
    push!(layers, fpad.(ac.pi.net, n_obs)...)
    push!(layers, fpad(ac.pi.mu_layer, n_obs))

    # Add action squash
    squash = squash_model(act_mins, act_maxs)
    push!(layers, fpad.(squash.layers, n_obs)...)
    step_down = length(layers)

    return ExtendedNetwork(nnet=Network(layers), step_up=step_up, step_down=step_down)
end

"""
Convert actor-critic object to extended neural network representing ensemble of values.
"""
function values_network(ac::GlobalResult; act_mins::Vector{Float64}, act_maxs::Vector{Float64})
    # Compute policy network
    network = policy_network(ac; act_mins=act_mins, act_maxs=act_maxs)
    layers = network.nnet.layers

    # Replicate output
    sz = length(layers[end].bias)
    n_qs = length(ac.qs)
    W = repeat(Matrix{Float32}(I, sz, sz), n_qs)
    b = zeros(Float32, n_qs*sz)
    σ = NeuralVerification.Id()
    push!(layers, Layer(W, b, σ))

    # Concatenate critics
    for ls in zip((q -> q.q).(ac.qs)...)
        W = block_diag((l -> l.weight).(ls)...)
        b = vcat((l -> l.bias).(ls)...)
        σ = convert_f(ls[1].σ)
        push!(layers, Layer(W, b, σ))
    end
    return network
end

"""
Convert actor-critic object to extended neural network representing mean of ensemble.
"""
function mean_network(ac::GlobalResult; act_mins::Vector{Float64}, act_maxs::Vector{Float64}, s::Float64=0.0)
    network = values_network(ac; act_mins=act_mins, act_maxs=act_maxs)
    layers = network.nnet.layers

    sz = length(layers[end].bias)
    push!(layers, mean_layer(sz))

    # Compensation for reward shaping
    if s != 0.0
        for i in eachindex(layers)
            layer = layers[i]
            W = layer.weights
            if i > 1
                W = hcat(W, zeros(Float32, size(W, 1), 1))
            end
            W = vcat(W, zeros(Float32, 1, size(W, 2)))
            W[end, i == 1 ? 1 : end] = 1.0f0
            b = vcat(layer.bias, 0.0f0)
            layers[i] = Layer(W, b, layer.activation)
        end

        W = [1.0f0 -Float32(s)]
        b = [0.0f0]
        σ = NeuralVerification.Id()
        push!(layers, Layer(W, b, σ))
        network.scale = s
    end

    return network
end

"""
Convert actor-critic object to extended neural network representing spread of ensemble
(mean absolute deviation).
"""
function spread_network(ac::GlobalResult; act_mins::Vector{Float64}, act_maxs::Vector{Float64})
    network = values_network(ac; act_mins=act_mins, act_maxs=act_maxs)
    layers = network.nnet.layers

    # Distance to mean
    sz = length(layers[end].bias)
    M1 = Matrix{Float32}(I, sz, sz)
    M2 = Float32.(ones(sz, sz) / sz)
    W = M1 - M2
    b = zeros(Float32, sz)
    σ = NeuralVerification.Id()
    push!(layers, Layer(W, b, σ))

    push!(layers, abs_layers(sz)...)
    push!(layers, mean_layer(sz))
    return network
end

"""
    CrossSection

Cross-section type, defining linear mapping from reduced input space to network observation
space. Input can be specified as explicit matrix and bias or as functional mapping.

# Example
```
cs1 = CrossSection([1 -1; 0 1; 0 0], [0, 0, 5])
cs2 = CrossSection((x1, x2) -> (x1, -pi/2, -x2, pi/2, x1 + x2))
```
"""
mutable struct CrossSection
    W::Matrix{<:Real}
    b::Vector{<:Real}
end

function CrossSection(f::Function)
    unit = (n, i) -> [Float64(i == j) for j in 1:n]
    n = first(methods(f)).nargs - 1
    b = collect(f(zeros(n)...))
    W = reduce(hcat, collect(f(unit(n, i)...)) - b for i in 1:n)
    return CrossSection(W, b)
end

"""
    cross_section(network::ExtendedNetwork, cs::CrossSection, limits::NTuple{2, Vector{Float64}})

Produce cross-section of neural network to reduce input dimensionality.
"""
function cross_section(network::ExtendedNetwork, cs::CrossSection, limits::NTuple{2, Vector{Float64}})
    σ = NeuralVerification.Id()
    layer = [Layer(Float32.(cs.W), Float32.(cs.b), σ)]

    # Determine necessary step
    min_obs = compute_output(Network(layer), limits[1])
    max_obs = compute_output(Network(layer), limits[2])
    steps = -min.(min_obs, max_obs)

    # Duplicate and step network
    n_obs = length(cs.b)
    nnet = deepcopy(network.nnet)
    i1, i2 = network.step_up, network.step_down
    nnet.layers[i1].bias[1:n_obs] = Float32.(steps)
    nnet.layers[i2].bias[1:n_obs] = -Float32.(steps)

    # Add cross-section layer
    prepend!(nnet.layers, layer)

    return coalesce(nnet)
end
