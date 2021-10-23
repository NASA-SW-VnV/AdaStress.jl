"""
Collision avoidance system type. Partially overrides pilot intent if threat is detected.
"""
abstract type CAS end

"""
Simple rule-based collision avoidance system.
"""
Base.@kwdef mutable struct RuleBasedCAS <: CAS
	 hra::Symbol = :none  # horizontal resolution advisory
	 vra::Symbol = :none  # vertical resolution advisory
	 t_predict::R = 5.0   # prediction horizon [s]
	 d::R = 50.0          # lowest tolerable separation [ft]
	 psi_crit::R = 15.0   # determines reversal regions
end

reverse(ra::Symbol) = ra == :cw ? :ccw : ra == :ccw ? :cw : :none

"""
Determine relative heading region of aircraft for input to ruleset. Regions partition the
angle space defined in the `kernel` function, with horizontal velocity measured clockwise
from the +y direction. Intruder aircraft angle is flipped for symmetry.
"""
function region(psi::R, psi_crit::R; own::Bool=true)
	# reflect intruder angle
	xi = rad2deg(own ? psi : -psi)

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
    # can only resolve pairwise interactions
    for j in filter(j -> j != i, 1:length(acs))
        advise(cas, acs[i], acs[j], i, j)
    end
end

"""
Use rule-based CAS to generate horizontal and vertical resolution advisories for a pair of
aircraft. There is a general turning rule, two reversal rules, and a vertical rule. The
aircraft containing the CAS is referred to as the ownship, while the other aircraft is the
intruder.

General turn rule: if ownship is aimed to the right of the intruder, turn clockwise;
otherwise, turn counterclockwise.

Reversal rules (override general turn rule):
1. If ownship is aimed within `psi_crit` of the intruder and intruder is aimed
more than `psi_crit` from the ownship in the same direction, turn in the opposite direction.
2. If both aircraft are aimed within `psi_crit` of each other in the same direction, use id
number to break the tie, where the aircraft with the higher id has lower precendence and
must switch direction.

Vertical rule: If ownship is above the intruder, ascend; if ownship is below the intruder,
descend. If aircraft are at the same altitude, lower-precendence aircraft must descend.
"""
function advise(cas::RuleBasedCAS, ac1::Aircraft, ac2::Aircraft, i::Z, j::Z)
	# assess threat within prediction window
	threat = assess_threat(cas, ac1, ac2)
	if !threat
		cas.hra = cas.vra = :none
		return
	end

	# get regions and establish precedence
	state = kernel(ac1, ac2; scale=false)
	r1 = region(state[2], cas.psi_crit, own=true)
	r2 = region(state[4], cas.psi_crit, own=false)
    defer = i > j

	# general turning rule
    cas.hra = r1 in (:i, :I, :II) ? :cw : :ccw

	# reversal rules
	rule1_ccw = r1 in (:i,) && r2 in (:I, :II)
    rule1_cw = r1 in (:iv,) && r2 in (:III, :IV)
	rule2 = r1 in (:i, :iv) && r1 == r2 && defer
    if rule1_cw || rule1_ccw || rule2
        cas.hra = reverse(cas.hra)
    end

	# vertical rule
	z1, z2 = ac1.r[3], ac2.r[3]
    if z1 > z2
        cas.vra = :asc
    elseif z1 < z2
        cas.vra = :desc
    else
        cas.vra = defer ? :desc : :none
    end
end

"""
Return true if projected aircraft trajectories pass within critical distance of each other
during prediction window. Analytical calculation naively assumes constant velocity.
"""
function assess_threat(cas::RuleBasedCAS, ac1::Aircraft, ac2::Aircraft)
	Δr = ac1.r - ac2.r
	Δv = ac1.v - ac2.v

	a = dot(Δv, Δv)
	b = 2 * dot(Δr, Δv)
	c = dot(Δr, Δr) - cas.d^2
	D = b^2 - 4 * a * c
	D < 0 && return false

	ts = (-b .+ [1, -1] * sqrt(D)) / (2 * a) # times of closest approach (may be in past)
	threat = any(0 .< ts .<= cas.t_predict)
	return threat
end

vec(cas::CAS) = [cas.hra; cas.vra]
