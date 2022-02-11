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
Distance heuristic abstract type. Custom heuristics inherit this type and implement `reset!`
and `apply!` functions. Application can return scalar value or partially-applied function
transforming f(s,s') to f(s)(s').
"""
abstract type AbstractDistanceHeuristic end
reset!(::AbstractDistanceHeuristic) = nothing
apply!(::AbstractDistanceHeuristic, ::ASTMDP) = unimplemented()
(adh::AbstractDistanceHeuristic)(mdp::ASTMDP) = Functoid(apply!(adh, mdp))

"""
Gradient of conservative potential. Default and recommended.
"""
struct GradientHeuristic <: AbstractDistanceHeuristic end

function apply!(::GradientHeuristic, mdp::ASTMDP)
    d = distance(mdp.sim)
    return mdp′::ASTMDP -> d - distance(mdp′.sim)
end

"""
Minimum distance across episode. Warning: non-Markovian.
"""
Base.@kwdef mutable struct MinimumHeuristic <: AbstractDistanceHeuristic
    d_min::Float64 = Inf
end

function apply!(h::MinimumHeuristic, mdp::ASTMDP)
    h.d_min = min(h.d_min, distance(mdp.sim))
    return terminated(mdp) ? -h.d_min : 0.0
end

function reset!(h::MinimumHeuristic)
    h.d_min = Inf
end

"""
Final distance encountered. If MDP is episodic, this is equal to the minimum distance
(determined retroactively instead of accumulated as in MinimumHeuristic).
"""
struct FinalHeuristic <: AbstractDistanceHeuristic end
apply!(::FinalHeuristic, mdp::ASTMDP) = isterminal(mdp.sim) ? -distance(mdp.sim) : 0.0

"""
Null heuristic. Returns zero.
"""
struct NullHeuristic <: AbstractDistanceHeuristic end
apply!(::NullHeuristic, ::ASTMDP) = 0.0
