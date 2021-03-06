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
Job description sent to worker processes. Stores new policy to be rolled out.
"""
struct Master2Worker
    ac::MLPActorCritic      # policy
    n_steps::Int            # number of steps
    random_policy::Bool     # whether to follow policy
end

"""
Result sent from worker processes to master process. Stores data from rollouts.
"""
struct Worker2Master
    id::Int                                     # worker ID
    transition_tuples::Vector{NamedTuple}       # data from rollouts
end

"""
Worker-side task. Runs in loop, receiving policies and sending observed data.
"""
function simulation_task(jobs::RemoteChannel, results::RemoteChannel, env_fn::Function, max_ep_len::Int)

    # Initialize environment and collection
	Random.seed!(myid())
    env = env_fn()
    transition_tuples = []

    # Loop until end of program
    while true
        # Receive job from master process and initialize RNG
        job = take!(jobs)
        copy!(job.ac.pi.rng, Random.default_rng())

        empty!(transition_tuples)
        ep_ret, ep_len = 0.0, 0
        reset!(env)
        o = observe(env)
        for _ in 1:job.n_steps
            # Choose action
        	a = job.random_policy ? rand(actions(env); flat=true) : job.ac(o)

            # Step environment
            d = terminated(env)
            r = act!(env, a)
            o2 = observe(env)
            ep_ret += r
            ep_len += 1

            # Ignore done signal if due to overtime
            d = (ep_len > max_ep_len) ? false : d

            push!(transition_tuples, (o=o, a=a, r=r, o2=o2, d=d))
            o = o2

            # End of trajectory handling
            if d || ep_len > max_ep_len
                d, ep_ret, ep_len = false, 0.0, 0
                reset!(env)
                o = observe(env)
            end
        end

        # Send results to master process
        result = Worker2Master(myid(), transition_tuples)
        put!(results, result)
    end
end

"""
Master-side code. Delegates work and performs updates.
"""
function distributed_solve(sac::SAC, env_fn::Function; channel_size::Int=32)
    # Create remote channels
    jobs = RemoteChannel(()->Channel{Master2Worker}(channel_size));
    results = RemoteChannel(()->Channel{Worker2Master}(channel_size));

    # Initialize AC agent and auxiliary data structures
    test_env = env_fn()
    replay_buffer = ReplayBuffer(sac.obs_dim, sac.act_dim, sac.max_buffer_size)
    ac = MLPActorCritic(sac.obs_dim, sac.act_dim, sac.act_mins, sac.act_maxs,
        sac.hidden_sizes, sac.num_q, sac.activation, sac.rng)
    ac_targ = deepcopy(ac)
    alpha = [1.0]
    total_steps = sac.steps_per_epoch * sac.epochs

    # Enforce even work delegation
    steps_per_worker = Int(sac.update_every // nworkers())
    @assert sac.steps_per_epoch % sac.update_every == 0

    # Set up remote task
    for p in workers()
        remote_do(simulation_task, p, jobs, results, env_fn, sac.max_ep_len)
    end

    # Initialize displayed information and progress meter
    @debug "Solve" total_steps
    disptups = [(:score, []), (:qdiff, []), ((sym, []) for (sym, _) in sac.displays)...]
    p = Progress(total_steps)

    for t in sac.update_every:sac.update_every:total_steps

        # Delegate work
        random_policy = t <= sac.start_steps
        ac_cpu = deepcopy(ac)
        to_cpu!(ac_cpu)
        for _ in 1:nworkers()
            job = Master2Worker(ac_cpu, steps_per_worker, random_policy)
            put!(jobs, job)
        end

        # Actor-critic update (concurrent with rollouts)
        if t >= sac.update_after
            @debug "Updating models" t
            for _ in 1:sac.num_batches
                batch = sample_batch(replay_buffer, sac.batch_size)
                update(ac, ac_targ, batch, sac.q_optimizer, sac.pi_optimizer, sac.polyak,
                    sac.gamma, sac.alpha_optimizer, alpha, sac.target_entropy)
            end
        end

        # End of epoch handling
        epoch = (t - 1) ÷ sac.steps_per_epoch + 1
        if t % sac.steps_per_epoch == 0
            # Update display values
            dispvals = test_agent(ac, test_env, sac.displays, sac.max_ep_len, sac.num_test_episodes)
            for ((_, hist), val) in zip(disptups, dispvals)
                push!(hist, val)
            end

            # Log info about epoch
            @debug("Evaluation",
            	  alpha[],
            	  mean(mean.(params([q.q for q in ac.qs]))),
            	  mean(mean.(Flux.params([ac.pi.net, ac.pi.mu_layer, ac.pi.log_std_layer])))
            )
        end

        # Collect and order results (will block here)
        results_acc = []
        for _ in 1:nworkers()
        	push!(results_acc, take!(results))
        end
        sort!(results_acc, by=r->r.id)

        # Store in replay buffer
        for result in results_acc
            for tup in result.transition_tuples
                store!(replay_buffer, tup...)
            end
        end

        # Checkpointing
        if sac.save && t > sac.update_after && t % sac.save_every == 0
            checkpoint(ac, sac.save_dir, sac.max_saved)
        end

        # Progress meter
        if t > sac.update_after
            ProgressMeter.next!(p; showvalues=gen_showvalues(epoch, disptups))
        end
    end

    # Save display values and replay buffer
    info = Dict{String,Any}()
    for (sym, hist) in disptups
        info[String(sym)] = hist
    end
    info["replay_buffer"] = replay_buffer

    return ac, info
end
