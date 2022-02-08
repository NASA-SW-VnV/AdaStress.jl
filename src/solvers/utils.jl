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
Delete least significant keys if length exceeds `k`. Facilitates top-k priority queueing.
"""
function DataStructures.enqueue!(pq::PriorityQueue, key, value, k::Int64)
    pq[key] = value
    while length(pq) > k
        delete!(pq, first(keys(pq)))
    end
end

function replay!(mdp::Interface.AbstractASTMDP, result::LocalResult)
    as = getproperty(result, fieldnames(typeof(result))[1])
    Interface.reset!(mdp)
    for a in as
        Interface.act!(mdp, a)
    end
    return Interface.isevent(mdp.sim)
end

function replay!(mdp::Interface.AbstractASTMDP, result::GlobalResult)
    Interface.reset!(mdp)
    while !Interface.terminated(mdp)
        a = result(Interface.observe(mdp))
        Interface.act!(mdp, a)
    end
    return Interface.isevent(mdp.sim)
end

struct RandomPolicy end

function replay!(mdp::Interface.AbstractASTMDP, ::RandomPolicy)
    Interface.reset!(mdp)
    while !Interface.terminated(mdp)
        a = rand(Interface.actions(mdp))
        Interface.act!(mdp, a)
    end
    return Interface.isevent(mdp.sim)
end

struct NullPolicy end

function replay!(mdp::Interface.AbstractASTMDP, ::NullPolicy)
    Interface.reset!(mdp)
    while !Interface.terminated(mdp)
        a = zero(rand(Interface.actions(mdp)))
        Interface.act!(mdp, a)
    end
    return Interface.isevent(mdp.sim)
end
