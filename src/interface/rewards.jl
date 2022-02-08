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
AST core objective function abstract type. Defines fundamental reward function by combining
log probability, event bonus, and distance heuristic. Additional contributions to the
reward should be added by implementing `reward` function in GrayBox or BlackBox interface.
"""
abstract type AbstractCoreObjective end

"""
Default objective function. Sums components at each timestep.
"""
Base.@kwdef mutable struct WeightedObjective <: AbstractCoreObjective
    wl::Float64 = 1.0
    we::Float64 = 1.0
    wh::Float64 = 1.0
end

function (rf::WeightedObjective)(logprob::Float64, event::Float64, heuristic::Float64)
    return rf.wl * logprob + rf.we * event + rf.wh * heuristic
end

"""
Vector reward function. Maintains separate components to facilitate post-analysis and
enhanced learning methods.
"""
struct VectorObjective <: AbstractCoreObjective end

function (::VectorObjective)(logprob::Float64, event::Float64, heuristic::Float64)
    return (logprob, event, heuristic)
end

"""
Standard AST reward structure.
"""
Base.@kwdef mutable struct Reward <: AbstractReward
    marginalize::Bool                      = true
    heuristic::AbstractDistanceHeuristic   = GradientHeuristic()
    event_bonus::Float64                   = 0.0
    reward_function::AbstractCoreObjective = WeightedObjective()
end

"""
Internal handling of custom reward.
"""
custom_reward(sim::GrayBox, a::SampleAction) = Functoid(reward(sim, a.sample))
custom_reward(sim::BlackBox, ::SeedAction) = Functoid(reward(sim))
