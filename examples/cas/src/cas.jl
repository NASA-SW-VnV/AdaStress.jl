"""
Collision avoidance system type. Partially overrides pilot intent if threat is detected.
"""
abstract type CAS end

"""
Rule-based collision avoidance system.
"""
Base.@kwdef mutable struct RuleBasedCAS <: CAS
	 hra::Symbol = :none  # horizontal resolution advisory
	 vra::Symbol = :none  # vertical resolution advisory
	 t_predict::R = 5.0   # prediction horizon [s]
	 d::R = 50.0          # lowest tolerable separation [ft]
	 psi_crit::R = 15.0   # determines reversal regions
end

reverse(ra::Symbol) = ra == :cw ? :ccw : ra == :ccw ? :cw : :none

function region(psi, psi_crit, i)
	# reflect intruder angle
	xi = rad2deg(i == 1 ? psi : -psi)

	# discretize
	bins = [-180, -90 - psi_crit, -90, -90 + psi_crit, 0, 90]
	regs = [:IV, :iv, :i, :I, :II, :III]
	idx = searchsortedlast(bins, xi)
	return regs[idx]
end

function initialize(cas::RuleBasedCAS)
	cas.hra = :none
	cas.vra = :none
end

function update(cas::RuleBasedCAS, acs::Vector{Aircraft}, i::Z)
	ac1 = acs[i]
    for j in filter(j -> j != i, 1:length(acs))
	    ac2 = acs[j]
        advise(cas, ac1, ac2, i, j) # can only resolve pairwise interactions
    end
end

function advise(cas::RuleBasedCAS, ac1::Aircraft, ac2::Aircraft, i::Z, j::Z)
    #TODO: review

	# assess threat within prediction window
	threat = assess_threat(cas, ac1, ac2)
	if !threat
		cas.hra = cas.vra = :none
		return
	end

	# get regions
	state = kernel(ac1, ac2; scale=false)
	r1 = region(state[2], cas.psi_crit, 1)
	r2 = region(state[4], cas.psi_crit, 2)

	# clockwise turning rule
	hra = r1 in (:i, :I, :II) ? :cw : r1 in (:III, :IV, :iv) ? :ccw : none

	# reversal rules
	rule1 = (r1 in (:i,) && r2 in (:I, :II)) || (r1 in (:iv,) && r2 in (:III, :IV))
	rule2 = r1 in (:i, :iv) && r1 == r2 && i > j
	cas.hra = rule1 || rule2 ? reverse(hra) : hra

	# vertical rule
	z1, z2 = ac1.r[3], ac2.r[3]
	cas.vra = z1 > z2 ? :asc : z1 < z2 || i > j ? :desc : :none
end

function assess_threat(cas::RuleBasedCAS, ac1::Aircraft, ac2::Aircraft)
	Δr = ac1.r - ac2.r
	Δv = ac1.v - ac2.v

	a = dot(Δv, Δv)
	b = 2 * dot(Δr, Δv)
	c = dot(Δr, Δr) - cas.d^2
	D = b^2 - 4 * a * c
	D < 0 && return false

	ts = (-b .+ [1, -1] * sqrt(D)) / (2 * a)
	threat = any(0 .< ts .<= cas.t_predict)
	return threat
end

vec(cas::CAS) = [cas.hra; cas.vra]
