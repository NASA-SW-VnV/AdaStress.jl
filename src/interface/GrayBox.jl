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

# To be implemented by user.

"""
    GrayBox <: AbstractSimulation

Parent type for user simulation in which environment is exposed.
"""
abstract type GrayBox <: AbstractSimulation end

"""
    reset!(sim::GrayBox)

Reset simulation.
"""
reset!(sim::GrayBox)::Nothing = unimplemented()

"""
    environment(sim::GrayBox)

Return `Environment` object constructed by simulation. `Environment` is an alias of
`Dict{Symbol, Sampleable}`.
"""
environment(sim::GrayBox)::Environment = unimplemented()

"""
    observe(sim::GrayBox)

Return quasi-normalized observation of simulation. Observation should copy, not reference,
simulation state. If simulation is unobservable, leave unimplemented.
"""
observe(sim::GrayBox)::AbstractVector{<:Real} = unimplemented()

"""
    step(sim::GrayBox, x::EnvironmentValue)

Step simulation given an `EnvironmentValue` object. `EnvironmentValue` is an alias of
`Dict{Symbol, Any}`.
"""
step!(sim::GrayBox, x::EnvironmentValue)::Nothing = unimplemented()

"""
    isterminal(sim::GrayBox)

Check whether simulation has finished due to time limit or terminal state, independent of
failure status.
"""
isterminal(sim::GrayBox)::Bool = unimplemented()

"""
   isevent(sim::GrayBox)

Check whether simulation is in event (failure) state. For episodic simulation, check whether
event occurred during episode.
"""
isevent(sim::GrayBox)::Bool = unimplemented()

"""
    distance(sim::GrayBox)

Return custom metric of distance to event. For best results, metric should depend only on
current state. For episodic simulation, return minimum distance to event across entire
episode (miss distance).
"""
distance(sim::GrayBox)::Real = unimplemented()

"""
    flatten(distribution, value)

Flatten environment variable to quasi-normalized array.
"""
flatten(distribution::Any, value::Any)::AbstractVector{<:Real} = unimplemented()
flatten(d::Distribution{Univariate, Continuous}, v::Real) = [(v - mean(d)) / std(d)]
flatten(d::Uniform, v::Real) = [(2 * v - d.a - d.b) / (d.b - d.a)]

"""
    unflatten(distribution, array::AbstractVector{<:Real})

Reconstruct environment variable from quasi-normalized array.
"""
unflatten(distribution::Any, array::AbstractVector{<:Real})::Any = unimplemented()
unflatten(d::Distribution{Univariate, Continuous}, a::AbstractVector{<:Real}) = std(d) * a[] + mean(d)
unflatten(d::Uniform, a::AbstractVector{<:Real}) = (a[] * (d.b - d.a) + d.a + d.b) / 2

"""
    reward(sim::GrayBox, x::EnvironmentValue)

Advanced option: additional reward or heuristic. Returns scalar value or partially-applied
function, allowing efficient calculation of R(s,a,s') = R(s,a)(s'). Should not alter
simulation. See documentation for more details.
"""
reward(sim::GrayBox, x::EnvironmentValue) = 0.0
