"""
Encounter object. Specifies distribution of initial conditions.
"""
Base.@kwdef mutable struct Encounter
	n::Z                  = 2                      # number of aircraft
	acs::Vector{Aircraft} = @array n Aircraft()
    randomize::Bool       = true
	seed::Z               = 0
	rng::AbstractRNG      = MersenneTwister(seed)
	d_min::R              = 1000.0                 # resample threshold [ft]
	v_dist::Distribution  = Uniform(50.0, 150.0)   # ground speed distribution [ft/s]
	t_c::R                = 25.0                   # nominal collision time [s]
	r_c::Vector{R}        = [0.0; 0.0; 5000.0]     # nominal collision position [ft]
end

function initialize(enc::Encounter)
    rng = enc.randomize ? Random.default_rng() : deepcopy(enc.rng)
	d = -Inf
    # resample if any two aircraft are closer than threshold
	while d <= enc.d_min
		for ac in enc.acs
			v2 = rand(rng, enc.v_dist) * normalize(randn(rng, 2))
			ac.v = [v2; 0.0]
			ac.r = enc.r_c .- ac.v * enc.t_c
		end
		d = minimum(distance(pair...) for pair in subsets(enc.acs, 2))
	end
end
