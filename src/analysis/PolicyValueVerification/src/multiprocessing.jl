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

const TERMINATE = 0x0

"""
Make k workers available in total, spawning new processes only as necessary.
"""
get_workers(k::Int64) = addprocs(max(k + 1 - nprocs(), 0))

"""
Recursively build miniminal k-d tree that defines volumes proven to satisfy given condition
or its complement. At specificed depth levels, places children onto jobs queue instead of
refining, to maintain a balanced work distribution across processes.
"""
function refine!(cell::Cell, r::AbstractRefinery, jobs::RemoteChannel)
    cell.data.pid = myid()
    if needs_refinement(cell, r)
        split!(cell, child_data)
        cs = children(cell)

        if cell.data.depth in r.ks
            for c in cs
                c.parent = nothing      # detach to avoid copying entire tree onto queue
                c.data.detached = true
                put!(jobs, c)
            end
            remotecall_wait(add_todo, 1, length(cs)) # wait for acknowledgement
        else
            for c in cs
                refine!(c, r, jobs)
            end
            merge!(cell)
        end
    end
    return nothing
end

"""
Take a cell from the jobs queue, refine it (possibly spawning more jobs in the process), and
place it onto the results queue. Result may not be a well-formed tree; the worker does not
wait on detached children. The forest is stitched together by the main process after all
workers have completed. Workers increment a shared atomic counter when adding to the jobs
queue and decerement it when adding to the results queue. Workers break upon receiving a
termination signal (a cell with a null hash).
"""
function work(jobs::RemoteChannel, results::RemoteChannel, r::AbstractRefinery)
    while true
        cell = take!(jobs)
        cell.data.hash == TERMINATE && break
        refine!(cell, r, jobs)
        put!(results, cell)
        remote_do(add_done, 1, 1) # no wait required
    end
end

"""
Recursively reassemble forest, merging a cell with a dictionary of subtrees ordered by cell
hash.
"""
function merge!(cell::Cell, subtrees::Dict{UInt64, <:Cell})
    isleaf(cell) && return nothing
    for c in children(cell)
        # Repair bidirectional link
        c.parent = cell

        # If child is detached, search dictionary for computed version and merge.
        if c.data.detached
            cc = pop!(subtrees, c.data.hash)
            c.data = cc.data
            c.data.detached = false
            c.children = cc.children
        end
        merge!(c, subtrees)
    end
    merge!(cell)
    return nothing
end

"""
Empty results channel, perform merge, and verify correctness.
"""
function merge!(root::Cell, results::RemoteChannel)
    # Collects subtrees.
    forest = []
    while isready(results)
        push!(forest, take!(results))
    end

    # Creates hash set.
    subtrees = Dict(c.data.hash => c for c in forest)

    # Merges root with subtrees.
    rootc = pop!(subtrees, root.data.hash)
    root.children = rootc.children
    merge!(root, subtrees)

    # Verfies successful merge.
    @assert isempty(subtrees)
    @assert verify(root)

    return nothing
end

"""
Terminate worker processes.
"""
function terminate(jobs::RemoteChannel)
    signal = Cell(SVector{0, Float64}(), SVector{0, Float64}(), CellStatus(hash=TERMINATE))
    for _ in workers()
        put!(jobs, signal)
    end
end

"""
    refine_multiprocess!(root::Cell, r::AbstractRefinery)

Run multiprocess refinement.
"""
function refine_multiprocess!(root::Cell, r::AbstractRefinery)
    jobs = RemoteChannel(() -> Channel{Cell}(1_000_000))
    results = RemoteChannel(() -> Channel{Cell}(1_000_000))

    # Remotely execute work task.
    put!(jobs, root)
    TODO_COUNTER[] = 1
    for p in workers()
        remote_do(work, p, jobs, results, r)
    end

    # Block until there is no outstanding work, then send termination signal.
    while TODO_COUNTER[] > 0
    	sleep(0.1)
    end
    terminate(jobs)

    # Collect results and merge with root.
    merge!(root, results)
end
