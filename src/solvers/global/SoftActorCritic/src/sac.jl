"""
Compute loss for value function under current policy.
"""
function compute_loss_q(
	ac::MLPActorCritic,
	ac_targ::MLPActorCritic,
	data::NamedTuple,
	gamma::Float64,
	alpha::AbstractVector{Float32};
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
    alpha::AbstractVector{Float32};
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
function compute_loss_alpha(ac::MLPActorCritic, data::NamedTuple, alpha::AbstractVector{Float32}, target_entropy::Float64)
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

    # Transfer data to GPU
    data = (; zip(keys(data), gpu.(values(data)))...)

    # Gradient descent step for value networks
    loss_q = 0.0
    q_ps = params((q -> q.q).(ac.qs))
    q_gs = gradient(q_ps) do
        loss_q = compute_loss_q(ac, ac_targ, data, gamma, alpha)
        return loss_q
    end
    Flux.update!(q_optimizer, q_ps, q_gs)

    # Gradient descent step for policy network
    loss_pi = 0.0
    pi_ps = params([ac.pi.net, ac.pi.mu_layer, ac.pi.log_std_layer])
    pi_gs = gradient(pi_ps) do
        loss_pi = compute_loss_pi(ac, data, alpha)
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
TODO: Statistics can be gathered more efficiently from existing rollouts.
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
            a = ac(o, true)
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
    hidden_sizes::Vector{Int} = [100,100,100]       # dimensions of any hidden layers
    num_q::Int = 2                                  # size of critic ensemble
    activation::Function = SoftActorCritic.relu     # activation after each hidden layer

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

    # Testing
    num_test_episodes::Int = 100                    # number of test episodes
    displays::Vector{<:Tuple} = Tuple[]             # display values (list of tuples of
                                                    # name and function to apply to MDP
                                                    # after each trajectory)

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
    test_env = env_fn()
    ac = MLPActorCritic(sac.obs_dim, sac.act_dim, sac.act_mins, sac.act_maxs,
        sac.hidden_sizes, sac.num_q, sac.activation, sac.rng)
    ac_targ = deepcopy(ac)
    ac_cpu = to_cpu(ac)
    alpha = [1.0f0] |> gpu
    total_steps = sac.steps_per_epoch * sac.epochs
    mkpath(sac.save_dir)

    # Initialize displayed information and progress meter
    @debug "Solve" total_steps
    disp_tups = initialize(sac.displays)
    p = Progress(total_steps - sac.update_after)

    ep_ret, ep_len = 0.0, 0
    reset!(env)
    o = observe(env)
    for t in 1:total_steps
        # Choose action
    	random_policy = t <= sac.start_steps
    	a = random_policy ? rand(actions(env); flat=true) : ac_cpu(o)

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

        # Actor-critic update
        if t > sac.update_after && t % sac.update_every == 0
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
        if sac.save && t > sac.update_after && t % sac.save_every == 0
            checkpoint(ac_cpu, sac.save_dir, sac.max_saved)
        end

        # Progress meter
        if t > sac.update_after
            ProgressMeter.next!(p; showvalues=gen_showvalues(epoch, disp_tups))
        end
    end

    # Save display values and replay buffer
    info = Dict{String, Any}()
    for (sym, hist) in disp_tups
        info[String(sym)] = hist
    end

    return ac_cpu, info
end
