"""
Simulator object. Holds all relevant variables and interfaces with AdaStress.
"""
Base.@kwdef mutable struct Simulator <: AdaStress.GrayBox
	n::Z                        = 2
    step::Z                     = 0
    num_steps::Z                = 50
	dt::R                       = 1.0
	seed::Z                     = 0
    randomize::Bool             = true
	enc::Encounter              = Encounter(n=n, randomize=randomize, seed=seed)
	acs::Vector{Aircraft}       = @array n Aircraft()
	cass::Vector{<:CAS}         = @array n RuleBasedCAS()
	prs::Vector{PilotResponse}  = @array n PilotResponse()
	trs::Vector{<:Distribution} = @array n MvNormal([10.0, 3.0, 3.0])
	nmac::Tuple{R,R}            = (50.0, 15.0)
    reduce_symmetries::Bool     = true
    metrics::Metrics            = Metrics()
    logging::Bool               = false
    log::Dict{String,Any}       = Dict{String, Any}()
end

function AdaStress.reset!(sim::Simulator)
	sim.step = 0
	initialize(sim.enc)
	initialize.(sim.acs, sim.enc.acs)
	initialize.(sim.cass)
	initialize.(sim.prs)
    initialize(sim.metrics, sim.acs, sim.nmac)
    initialize(sim.log, sim)
end

function AdaStress.environment(sim::Simulator)
    AdaStress.Environment(Symbol("cmd_$i") => sim.trs[i] for i in 1:sim.n)
end

function AdaStress.observe(sim::Simulator)
    if sim.n == 2 && sim.reduce_symmetries
        kernel(sim.acs...)
    else
        reduce(vcat, observation.(sim.acs))
    end
end

function AdaStress.step!(sim::Simulator, x::AdaStress.EnvironmentValue)
	sim.step += 1
	cmds = [Command(x[Symbol("cmd_$i")]) for i in 1:sim.n]
    update.(sim.cass, Ref(sim.acs), 1:sim.n)
    update.(sim.prs, cmds, sim.cass)
	update.(sim.acs, (pr -> pr.cmd).(sim.prs), sim.dt)
    update(sim.metrics, sim.acs, sim.nmac)
	update(sim.log, sim)
end

AdaStress.isterminal(sim::Simulator) = sim.step >= sim.num_steps

AdaStress.isevent(sim::Simulator) = sim.metrics.d["is_nmac"]

AdaStress.distance(sim::Simulator) = sim.metrics.d["d"]

AdaStress.flatten(dist::MvNormal, val::Vector{<:Real}) = sqrt(invcov(dist)) * val

AdaStress.unflatten(dist::MvNormal, val::Vector{<:Real}) = sqrt(cov(dist)) * val
