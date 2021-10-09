"""
Replay buffer.
"""
mutable struct ReplayBuffer
    obs_buf::Matrix{Float32}        # stores observations
    act_buf::Matrix{Float32}        # stores actions
    rew_buf::Vector{Float32}        # stores rewards
    obs2_buf::Matrix{Float32}       # stores post-action observations
    done_buf::Vector{Float32}       # stores termination signal
    ptr::Int                        # current free index
    size::Int                       # current size
    max_size::Int                   # maximum size
end

function ReplayBuffer(obs_dim::Int, act_dim::Int, max_size::Int)
    obs_buf = zeros(Float32, obs_dim, max_size)
    act_buf = zeros(Float32, act_dim, max_size)
    rew_buf = zeros(Float32, max_size)
    obs2_buf = zeros(Float32, obs_dim, max_size)
    done_buf = zeros(Float32, max_size)
    ptr = 1
    size = 0
    return ReplayBuffer(obs_buf, act_buf, rew_buf, obs2_buf, done_buf, ptr, size, max_size)
end

"""
Store data from one step of simulation.
"""
function store!(
    buf::ReplayBuffer,
    obs::Vector{<:Real},
    act::Vector{<:Real},
    rew::Real,
    next_obs::Vector{<:Real},
    done::Bool
)
    buf.obs_buf[:,buf.ptr] = obs
    buf.act_buf[:,buf.ptr] = act
    buf.rew_buf[buf.ptr] = rew
    buf.obs2_buf[:,buf.ptr] = next_obs
    buf.done_buf[buf.ptr] = done
    buf.ptr = mod1(buf.ptr + 1, buf.max_size)
    buf.size = min(buf.size + 1, buf.max_size)
    return buf.size
end

"""
Randomly sample from replay buffer.
"""
function sample_batch(buf::ReplayBuffer, batch_size::Int=32)
    idxs = rand(1:buf.size, batch_size)
    batch = (
        obs=buf.obs_buf[:,idxs],
        act=buf.act_buf[:,idxs],
        rew=buf.rew_buf[idxs],
        obs2=buf.obs2_buf[:,idxs],
        done=buf.done_buf[idxs]
    )
    return batch
end

"""
Lazily clear replay buffer.
"""
function clear!(buf::ReplayBuffer)
    buf.ptr = 1
    buf.size = 0
    return nothing
end
