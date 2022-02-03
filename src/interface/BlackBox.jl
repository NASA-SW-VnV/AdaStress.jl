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

# To be implemented by user.

"""
    BlackBox <: AbstractSimulation

Parent type for user simulation in which environment is not exposed.
"""
abstract type BlackBox <: AbstractSimulation end

"""
    reset!(sim::BlackBox)

Reset simulation.
"""
reset!(sim::BlackBox) = unimplemented()

"""
    observe(sim::BlackBox)

Return quasi-normalized observation of simulation. Observation should copy, not reference,
simulation state. If simulation is unobservable, leave unimplemented.
"""
observe(sim::BlackBox)::AbstractVector{<:Real} = unimplemented()

"""
    step([rng::AbstractRNG], sim::BlackBox)

Step simulation and return log probability of transition. If simulation uses global RNG,
the one-argument step! should be implemented. If simulation requires non-global RNG, the
two-argument function should be implemented instead.
"""
step!(sim::BlackBox)::Real = unimplemented()
step!(rng::AbstractRNG, sim::BlackBox)::Real = step!(sim)

"""
    isterminal(sim::BlackBox)

Check whether simulation has finished due to time limit or terminal state, independent of
failure status.
"""
isterminal(sim::BlackBox)::Bool = unimplemented()

"""
   isevent(sim::BlackBox)

Check whether simulation is in event (failure) state. For episodic simulation, check whether
event occurred during episode.
"""
isevent(sim::BlackBox)::Bool = unimplemented()

"""
    distance(sim::BlackBox)

Return custom metric of distance to event. For best results, metric should depend only on
current state. For episodic simulation, return minimum distance to event across entire
episode (miss distance).
"""
distance(sim::BlackBox)::Real = unimplemented()

"""
    reward(sim::BlackBox)

Advanced option: additional reward or heuristic. Returns scalar value or partially-applied
function, allowing efficient calculation of R(s,s') = R(s)(s'). Should not alter simulation.
See documentation for more details.
"""
reward(sim::BlackBox) = 0.0
