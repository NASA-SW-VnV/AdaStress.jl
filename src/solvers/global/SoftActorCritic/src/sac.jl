# ******************************************************************************************
# Notices:
#
# Copyright © 2022 United States Government as represented by the Administrator of the
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
Compute loss for value function under current policy.
"""
function compute_loss_q(
	ac::MLPActorCritic,
	ac_targ::MLPActorCritic,
	data::NamedTuple,
	gamma::Float64,
	alpha;
    average::Bool = true
)
	o, a, r, o2, d = data
    qs = [q(o, a) for q in ac.qs]
    a2, logp_a2 = ac.pi(o2)

    # Target q-values
    qs_pi_targ = [q(o2, a2) for q in ac_targ.qs]
    q_pi_targ = average ? sum(qs_pi_targ) / length(qs_pi_targ) : min.(qs_pi_targ...)
    backup = r .+ Float32(gamma) .* (1.0f0 .- d) .* (q_pi_targ .- alpha .* logp_a2)

    # MSE loss against Bellman backup
    loss_qs = [mean((q .- backup).^2) for q in qs]
    loss_q = sum(loss_qs)
    return loss_q
end

"""
Compute loss for current policy.
"""
function compute_loss_pi(
    ac::MLPActorCritic,
    data::NamedTuple,
    alpha;
    average::Bool = true
)
    o = data.obs
    pi, logp_pi = ac.pi(o)
    qs_pi = [q(o, pi) for q in ac.qs]
    q_pi = average ? sum(qs_pi) / length(qs_pi) : min.(qs_pi...)

    # Entropy-regularized policy loss
    loss_pi = mean(alpha .* logp_pi .- q_pi)
    return loss_pi
end

"""
Compute loss for current alpha.
"""
function compute_loss_alpha(ac::MLPActorCritic, data::NamedTuple, alpha, target_entropy::Float64)
    o = data.obs
    _, logp_pi = ac.pi(o)
    loss_alpha = mean(-1.0f0 .* alpha .* (logp_pi .+ Float32(target_entropy)))
    return loss_alpha
end

"""
Update policy with observations.
"""
function update(
	ac::MLPActorCritic,
	ac_targ::MLPActorCritic,
	data::NamedTuple,
	q_optimizer::Any,
	pi_optimizer::Any,
	polyak::Float64,
	gamma::Float64,
	alpha_optimizer::Any,
	alpha::AbstractVector{Float32},
	target_entropy::Float64
)
    average = length(ac.qs) > 2 # if ensemble is size 2, take min, otherwise take mean

    # Transfer data to GPU
    data = (; zip(keys(data), dev.(values(data)))...)

    # Gradient descent step for value networks
    loss_q = 0.0
    q_ps = params((q -> q.q).(ac.qs))
    q_gs = gradient(q_ps) do
        loss_q = compute_loss_q(ac, ac_targ, data, gamma, alpha; average=average)
        return loss_q
    end
    Flux.update!(q_optimizer, q_ps, q_gs)

    # Gradient descent step for policy network
    loss_pi = 0.0
    pi_ps = params([ac.pi.net, ac.pi.mu_layer, ac.pi.log_std_layer])
    pi_gs = gradient(pi_ps) do
        loss_pi = compute_loss_pi(ac, data, alpha; average=average)
        return loss_pi
    end
    Flux.update!(pi_optimizer, pi_ps, pi_gs)

    # Gradient descent step for alpha
    loss_alpha = 0.0
    alpha_ps = params(alpha)
    alpha_gs = gradient(alpha_ps) do
        loss_alpha = compute_loss_alpha(ac, data, alpha, target_entropy)
        return loss_alpha
    end
    Flux.update!(alpha_optimizer, alpha_ps, alpha_gs)

    # Update target networks with in-place Polyak averaging
    for (dest, src) in zip(params((q -> q.q).(ac_targ.qs)), params((q -> q.q).(ac.qs)))
        dest .= polyak .* dest .+ (1.0 .- polyak) .* src
    end

    # Logging
    @debug "Losses" loss_q loss_pi loss_alpha
end

"""
Test current policy and generate display statistics.
"""
function test_agent(
    ac::MLPActorCritic,
    test_env::CommonRLInterface.AbstractEnv,
    displays::Vector{<:Tuple},
    max_ep_len::Int,
    num_test_episodes::Int
)

    # Initialize collections for displayed values
    rets, stdevs = Float32[], Float32[]
    dispvals = [[] for _ in displays]

    # Perform rollouts
    for _ in 1:num_test_episodes

        # Single trajectory from deterministic (mean) policy
        d, ep_ret, ep_len = false, 0.0, 0
        reset!(test_env)
        o = observe(test_env)
        os, as = Vector{Float32}[], Vector{Float32}[]
        while !(d || ep_len > max_ep_len)
            a = ac(o)
            push!(os, o)
            push!(as, a)
            d = terminated(test_env)
            r = act!(test_env, a)
            o = observe(test_env)
            ep_ret += r
            ep_len += 1
        end

        # Batch process q-values
        O = reduce(hcat, os)
        A = reduce(hcat, as)
        qs = [q(O, A) for q in ac.qs]
        Q = reduce(hcat, qs)
        append!(stdevs, vec(std(Q, dims=2)))

        # Compute custom statistics and add to collections
        push!(rets, ep_ret)
        push!.(dispvals, f(test_env) for (_, f) in displays)
    end

    # Average displayed values
    dispvals = [rets, stdevs, dispvals...]
    dispvals_avg = mean.(dispvals)
    return dispvals_avg
end

"""
Soft actor-critic algorithm.
"""
Base.@kwdef mutable struct SAC <: GlobalSolver
    # Environment
    obs_dim::Int                                    # dimension of observation space
    act_dim::Int                                    # dimension of action space
    act_mins::Vector{Float64}                       # minimum values of actions
    act_maxs::Vector{Float64}                       # maximum values of actions
    gamma::Float64 = 0.999                          # discount factor

    # Replay buffer
    max_buffer_size::Int = 100000                   # maximum number of timesteps in buffer
    buffer::ReplayBuffer = ReplayBuffer(			# replay buffer
  		obs_dim, act_dim, max_buffer_size)

    # Actor-critic network
    ac::Union{MLPActorCritic, Nothing} = nothing    # actor-critic object
    hidden_sizes::Vector{Int} = [100,100,100]       # dimensions of any hidden layers
    num_q::Int = 2                                  # size of critic ensemble
    activation::Function = SoftActorCritic.relu     # activation after each hidden layer
    linearized::Bool = false                        # linearized policy squashing

    # Training
    q_optimizer::Any = AdaBelief(1e-4)              # optimizer for value networks
    pi_optimizer::Any = AdaBelief(1e-4)             # optimizer for policy network
    alpha_optimizer::Any = AdaBelief(1e-4)          # optimizer for alpha
    batch_size::Int = 64                            # size of each update to networks
    epochs::Int = 200                               # number of epochs
    steps_per_epoch::Int = 200                      # steps of simulation per epoch
    start_steps::Int = 1000                         # steps before following policy
    max_ep_len::Int = 50                            # maximum number of steps per episode
    update_after::Int = 300                         # steps before networks begin to update
    update_every::Int = 50                          # steps between updates
    num_batches::Int = update_every                 # number of batches per update
    polyak::Float64 = 0.995                         # target network averaging parameter
    target_entropy::Float64 = -act_dim              # target entropy (default is heuristic)
    rng::AbstractRNG = Random.GLOBAL_RNG            # random number generator
    use_gpu::Bool = true                            # use GPU if available

    # Testing
    num_test_episodes::Int = 100                    # number of test episodes
    displays::Vector{Tuple} = Tuple[]               # display values (list of tuples of
                                                    # name and function to apply to MDP
                                                    # after each trajectory)
    info::Dict{String, Any} = Dict{String, Any}()   # accumulated stats
    progress::Bool = true                           # show progress bar

    # Checkpointing
    save::Bool = false                              # to enable checkpointing
    save_every::Int = 10000                         # steps between checkpoints
    save_dir::String = DEFAULT_SAVE_DIR             # directory to save checkpoints
    max_saved::Int = 0                              # maximum number of checkpoints; set to
                                                    # zero or negative for unlimited
end

function Solvers.solve(sac::SAC, env_fn::Function)
    # Initialize AC agent and auxiliary data structures
    env = env_fn()
    env.flatten = true # TODO: infer automatically
    test_env = env_fn()

    set_gpu_status(sac.use_gpu)
    if sac.ac === nothing
        sac.ac = MLPActorCritic(sac.obs_dim, sac.act_dim, sac.act_mins, sac.act_maxs,
            sac.hidden_sizes, sac.num_q, sac.activation, sac.rng, sac.linearized)
    end
    ac = sac.ac
    ac_targ = deepcopy(ac)
    ac_cpu = to_cpu(ac)
    alpha = [1.0f0] |> dev
    total_steps = sac.steps_per_epoch * sac.epochs
    mkpath(sac.save_dir)

    # Initialize displayed information and progress meter
    @debug "Solve" total_steps
    disp_tups = initialize(sac.displays)
    p = Progress(total_steps - sac.update_after; enabled=sac.progress)

    ep_ret, ep_len = 0.0, 0
    reset!(env)
    o = observe(env)
    for t in 1:total_steps
        # Choose action
    	random_policy = t <= sac.start_steps
    	a = random_policy ? rand(actions(env)) : ac_cpu(o)

        # Step environment
        d = terminated(env)
        r = act!(env, a)
        o2 = observe(env)
        ep_ret += r
        ep_len += 1

        # Ignore done signal if due to overtime
        d = (ep_len > sac.max_ep_len) ? false : d

        store!(sac.buffer, o, a, r, o2, d)
        o = o2

        # End of trajectory handling
        if d || ep_len > sac.max_ep_len
            ep_ret, ep_len = 0.0, 0
            reset!(env)
            o = observe(env)
        end

        t <= sac.update_after && continue

        # Actor-critic update
        if t % sac.update_every == 0
            @debug "Updating models" t
            for b in 1:sac.num_batches
                @debug "Batch" b
                batch = sample_batch(sac.buffer, sac.batch_size)
                update(ac, ac_targ, batch, sac.q_optimizer, sac.pi_optimizer, sac.polyak,
                    sac.gamma, sac.alpha_optimizer, alpha, sac.target_entropy)
            end
            ac_cpu = to_cpu(ac)
        end

        # End of epoch handling
        epoch = (t - 1) ÷ sac.steps_per_epoch + 1
        if t % sac.steps_per_epoch == 0
            # Update display values
            disp_vals = test_agent(ac_cpu, test_env, sac.displays, sac.max_ep_len, sac.num_test_episodes)
            update!(disp_tups, disp_vals)

            # Log info about epoch
            @debug("Evaluation",
                alpha,
                mean(mean.(params([q.q for q in ac_cpu.qs]))),
                mean(mean.(params([ac_cpu.pi.net, ac_cpu.pi.mu_layer, ac_cpu.pi.log_std_layer])))
            )
        end

        # Checkpointing
        if sac.save && t % sac.save_every == 0
            checkpoint(ac_cpu, sac.save_dir, sac.max_saved)
        end

        # Progress meter
        sac.progress && ProgressMeter.next!(p; showvalues=gen_showvalues(epoch, disp_tups))
    end

    # Save display values
    for (sym, hist) in disp_tups
        k = String(sym)
        if k in keys(sac.info)
            append!(sac.info[k], hist)
        else
            sac.info[k] = hist
        end
    end

    ac_cpu.pi.rng_gpu = nothing
    return ac_cpu, sac.info
end
