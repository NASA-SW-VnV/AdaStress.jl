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

"""
Simple uninformed Monte Carlo search. Should be used as a baseline for other local solvers,
in terms of runtime performance and overall efficacy.
"""
module MonteCarloSearch

export
    MCS,
    solve

using ..Solvers
import ..Solvers.solve

using CommonRLInterface
using DataStructures: PriorityQueue, enqueue!
using ProgressMeter

"""
    MCS <: LocalSolver

Monte Carlo search solver.
"""
Base.@kwdef mutable struct MCS <: LocalSolver
    num_iterations::Int64 = 1000
    top_k::Int64          = 10
end

"""
    MCSResult <: LocalResult

Monte Carlo search result.
"""
mutable struct MCSResult <: LocalResult
    path::Vector
end

function Solvers.solve(mcs::MCS, env_fn::Function)
    mdp = env_fn()
    A = typeof(rand(actions(mdp)))
    best_paths = PriorityQueue()

    @showprogress for _ in 1:mcs.num_iterations
        reset!(mdp)
        path = A[]
        r = 0.0
        d = false
        while !d
            d = terminated(mdp)
            a = rand(actions(mdp))
            push!(path, a)
            r += act!(mdp, a)
            d && enqueue!(best_paths, MCSResult(path), r, mcs.top_k)
        end
    end

    return best_paths
end

end
