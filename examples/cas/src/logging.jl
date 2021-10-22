
STATIC_PROPERTIES = [:n, :num_steps, :dt, :seed, :randomize, :nmac]

function initialize(log::Dict{String,Any}, sim::Simulator)
    sim.logging || return

    for p in STATIC_PROPERTIES
        log["$p"] = getproperty(sim, p)
    end

    log["t"] = [sim.step * sim.dt]

    for i in 1:sim.n
    	log["ac_$i"] = [vec(sim.acs[i])]
    	log["cas_$i"] = [vec(sim.cass[i])]
    end

    for (k, v) in sim.metrics.d
        log[k] = [v]
    end
end

function update(log::Dict{String,Any}, sim::Simulator)
    sim.logging || return
    push!(log["t"], sim.step * sim.dt)

    for i in 1:sim.n
    	push!(log["ac_$i"], vec(sim.acs[i]))
    	push!(log["cas_$i"], vec(sim.cass[i]))
    end

    for (k, v) in sim.metrics.d
        push!(log[k], v)
    end
end
