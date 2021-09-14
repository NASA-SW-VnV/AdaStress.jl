
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
Translates regular Julia activation functions to nnet functions.
Limited to identity and relu. Attempts to infer anonymous functions.
"""
function func_translate(f::Function)
    fsym = Symbol(f)
    if fsym == :identity
        nsym = :Id
    elseif fsym == :relu
        nsym = :ReLU
    else
        p1 = f(1.0)
        m1 = f(-1.0)
        if p1 == 1.0 && m1 == -1.0
            nsym = :Id
            @warn "Inferred anonymous function as identity."
        elseif p1 == 1.0 && m1 == 0.0
            nsym = :ReLU
            @warn "Inferred anonymous function as relu."
        else
            error("Unable to translate or infer $f.")
        end
    end
    return getproperty(NeuralVerification, nsym)()
end

"""
Pads front of matrix with zeros.
"""
function fpad(M::Matrix{<:Real}, dims::Tuple{Int64,Int64})
    i, j = size(M)
    ip, jp = dims
    t = eltype(M)
    Mp = hcat(zeros(t, i + ip, jp), vcat(zeros(t, ip, j), M))
    return Mp
end

"""
Pads front of vector with zeros.
"""
function fpad(v::Vector{<:Real}, sz::Int64)
    t = eltype(v)
    vp = vcat(zeros(t, sz), v)
    return vp
end

"""
Pads front of matrix with identity.
"""
function fpad_id(M::Matrix{<:Real}, sz::Int64)
    Mp = fpad(M, (sz, sz))
    t = eltype(M)
    Mp[1:sz,1:sz] = Matrix{t}(I, sz, sz)
    return Mp
end

"""
Pads front of Flux.Dense layer and converts to nnet layer.
"""
function fpad(l::Dense, sz::Int64)
    W = fpad_id(l.W, sz)
    b = fpad(l.b, sz)
    σ = func_translate(l.σ)
    return Layer(W, b, σ)
end

"""
Constructs block diagonal matrix from matrix arguments.
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
Converts policy from actor-critic object to neural network.
Represents concatenation and squashing with configurations of ReLUs.
"""
function policy_network(ac::GlobalResult; act_mins::Vector{Float64}, act_maxs::Vector{Float64})
    # Dimensions and parameters
    n_obs = size(ac.pi.net.layers[1].W, 2)
    n_act = size(ac.pi.mu_layer.W, 1)
    steps = zeros(n_obs)
    layers = []

    # Adds duplication layer and step-up.
    M = Matrix{Float32}(I, n_obs, n_obs)
    W = vcat(M, M)
    b = zeros(Float32, 2*n_obs)
    b[1:n_obs] .+= Float32.(steps)
    σ = func_translate(identity)
    push!(layers, Layer(W, b, σ))
    step_up = length(layers)

    # Adds policy net.
    push!(layers, fpad.(ac.pi.net, n_obs)...)
    push!(layers, fpad(ac.pi.mu_layer, n_obs))

    # Adds action squash and state step-down.
    M = Matrix{Float32}(I, n_obs + n_act, n_obs + n_act)
    W = vcat(M, M[end-n_act+1:end,:])
    b = Float32.(vcat(zeros(n_obs), -act_mins, -act_maxs))
    σ = func_translate(relu)
    push!(layers, Layer(W, b, σ))

    M = Matrix{Float32}(I, n_obs + n_act, n_obs + n_act)
    W = hcat(M, -M[:,end-n_act+1:end])
    b = Float32.(vcat(-steps, act_mins))
    σ = func_translate(identity)
    push!(layers, Layer(W, b, σ))
    step_down = length(layers)

    # Creates extended network.
    nnet = Network(layers)
    return ExtendedNetwork(nnet=nnet, step_up=step_up, step_down=step_down)
end

"""
Converts actor-critic object to extended neural network representing ensemble of values.
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
    σ = func_translate(identity)
    push!(layers, Layer(W, b, σ))

    # Concatenate critics
    for ls in zip((q -> q.q).(ac.qs)...)
        W = block_diag((l -> l.W).(ls)...)
        b = vcat((l -> l.b).(ls)...)
        σ = func_translate(ls[1].σ)
        push!(layers, Layer(W, b, σ))
    end
    return network
end

"""
Converts actor-critic object to extended neural network representing mean of ensemble.
"""
function mean_network(ac::GlobalResult; act_mins::Vector{Float64}, act_maxs::Vector{Float64}, s::Float64=0.0)
    network = values_network(ac; act_mins=act_mins, act_maxs=act_maxs)
    layers = network.nnet.layers

    # Mean layer
    sz = length(layers[end].bias)
    W = Float32.(ones(1, sz) / sz)
    b = zeros(Float32, 1)
    σ = func_translate(identity)
    push!(layers, Layer(W, b, σ))

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
        σ = func_translate(identity)
        push!(layers, Layer(W, b, σ))
        network.scale = s
    end

    return network
end

"""
Converts actor-critic object to extended neural network representing spread of ensemble
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
    σ = func_translate(identity)
    push!(layers, Layer(W, b, σ))

    # Absolute value (two layer computation)
    M = Matrix{Float32}(I, sz, sz)
    W = vcat(M, -M)
    b = zeros(Float32, 2 * sz)
    σ = func_translate(relu)
    push!(layers, Layer(W, b, σ))

    W = hcat(M, M)
    b = zeros(Float32, sz)
    σ = func_translate(identity)
    push!(layers, Layer(W, b, σ))

    # Mean layer
    W = Float32.(ones(1, sz) / sz)
    b = zeros(Float32, 1)
    σ = func_translate(identity)
    push!(layers, Layer(W, b, σ))

    return network
end

"""
Cross-section type.
Defines simple mapping from reduced input space (:x1, :x2, ...) to network observation space.
	Example: cs = CrossSection([:x1, -pi/2, :x2, pi/2, :x2])
"""
CrossSection = Vector{Union{<:Real, Symbol}}

"""
Linear cross-section type.
Defines more general mapping.
	Example: W = [1 -1; 0 1; 0 0]; b = [0, 0, 5];
"""
mutable struct LinearCrossSection
    W::Matrix{<:Real}
    b::Vector{<:Real}
end

"""
Parses variables and creates cross-section linear mapping.
"""
function linearize(cs::CrossSection)
    vars = sort(unique(filter(el -> el isa Symbol, cs)))
    W = hcat(([el == var for el in cs] for var in vars)...)
    b = [el isa Real ? el : 0.0 for el in cs]
    return LinearCrossSection(W, b)
end

"""
Produces cross-section of neural network to reduce input dimensionality.
"""
function cross_section(network::ExtendedNetwork, cs::CrossSection, limits::NTuple{2, Vector{Float64}})
    return cross_section(network, linearize(cs), limits)
end

function cross_section(network::ExtendedNetwork, lcs::LinearCrossSection, limits::NTuple{2, Vector{Float64}})
    σ = NeuralVerification.Id()
    layer = [Layer(Float32.(lcs.W), Float32.(lcs.b), σ)]

    # Determines necessary step.
    min_obs = compute_output(Network(layer), limits[1])
    max_obs = compute_output(Network(layer), limits[2])
    steps = -min.(min_obs, max_obs) # necessary for general cross-section

    # Duplicates and steps network.
    n_obs = length(lcs.b)
    nnet = deepcopy(network.nnet)
    i1, i2 = network.step_up, network.step_down
    nnet.layers[i1].bias[1:n_obs] = Float32.(steps)
    nnet.layers[i2].bias[1:n_obs] = -Float32.(steps)

    # Adds cross-section layer.
    prepend!(nnet.layers, layer)

    return nnet
end
