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
