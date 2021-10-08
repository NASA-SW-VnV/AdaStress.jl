
"""
Recursively build miniminal k-d tree that defines volumes proven to satisfy given condition
or its complement. At specificed depth levels, places children onto jobs queue instead of
refining, to maintain an even work distribution across processes.
"""
function refine!(cell::Cell, r::AbstractRefinery, jobs::RemoteChannel)
    if needs_refinement(cell, r)
        split!(cell, child_data)
        cs = children(cell)

        # If cell is at detachment depth, children are detached and
        # placed onto jobs queue. Detachment is necessary to avoid
        # copying entire tree onto queue through parent references.
        if cell.data.depth in r.ks
            for c in cs
                c.parent = nothing
                c.data.detached = true
                put!(jobs, c)
            end
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
workers have completed. Each worker indicates its idle status by setting an index in a
shared array, and breaks upon receiving a termination signal (a cell with a 0x0 hash).
"""
function work(jobs::RemoteChannel, results::RemoteChannel, idle::SharedArray, r::AbstractRefinery)
    while true
        cell = take!(jobs)
        cell.data.hash == 0x0 && break
        idle[myid()] = false
        refine!(cell, r, jobs)
        put!(results, cell)
        idle[myid()] = true
    end
    return nothing
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
Terminate idle worker processes.
"""
function terminate(jobs::RemoteChannel)
    signal = Cell(SVector(0.0), SVector(0.0), CellStatus(hash=0x0))
    for p in workers()
        put!(jobs, signal)
    end
end

"""
    refine_multiprocess!(root::Cell, r::AbstractRefinery)

Run multiprocess refinement.
"""
function refine_multiprocess!(root::Cell, r::AbstractRefinery)
    # Remotely execute work task.
    put!(jobs, root)
	idle = SharedArray{Bool,1}((nprocs(),))
    for p in workers()
        remote_do(work, p, jobs, results, idle, r)
    end

    # Block until all workers report being idle, then send termination signal.
    #TODO: consider replacing idle array with global counter
    while !all(idle[2:end]) || isready(jobs)
    	sleep(1.0)
    end
    terminate(jobs)

    # Collect results and merge with root.
    merge!(root, results)
end
