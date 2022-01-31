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

const TODO_COUNTER = Threads.Atomic{Int64}(0)
const DONE_COUNTER = Threads.Atomic{Int64}(0)

add_todo(n::Int64=1) = Threads.atomic_add!(TODO_COUNTER, n)

function add_done(n::Int64=1)
    Threads.atomic_add!(DONE_COUNTER, n)
    Threads.atomic_sub!(TODO_COUNTER, n)
end

"""
Cell information pertaining to proof and processing status.
"""
Base.@kwdef mutable struct CellStatus
    depth::Int64 = 0
    detached::Bool = false
    hash::UInt64 = hash(0)
    member::Bool = false
    proven::Bool = false
    pid::Int64 = 0
end

"""
    BinaryRefinery <: AbstractRefinery

Binary refinement process.
"""
Base.@kwdef struct BinaryRefinery <: AbstractRefinery
    network::Network
    solver = ReluVal(max_iter=100)
    val::Float64 = 0.0                # boundary value
    output = HalfSpace([1.0], val)    # a⋅x ≤ b (region of interest)
    coutput = HalfSpace([-1.0], -val) # a⋅x ≥ b (complement region)
    tol::Float64 = 0.1                # maximum unproven length scale
    ks::Vector{Int64} = []            # detachment depths
end

"""
    IntervalRefinery <: AbstractRefinery

Interval refinement process.
"""
Base.@kwdef struct IntervalRefinery <: AbstractRefinery
    network::Network
    solver = ReluVal(max_iter=100)
    interval::Vector{Float64} = [-1.0, 1.0]
    output = Hyperrectangle(low=interval[1:1], high=interval[2:2])
    tol::Float64 = 0.1
    ks::Vector{Int64} = []
end

"""
Determine whether condition can be proven true or false over entire cell
or if further refinement is needed.
"""
function needs_refinement(cell::Cell, r::AbstractRefinery)
    # Checks status.
    cell.data.proven && return false

    # Checks tolerance.
    w = cell.boundary.widths
    maximum(w) < r.tol && return false

    # Determines corresponding input set.
    l = cell.boundary.origin
    h = l + w
    input = Hyperrectangle(low=convert(Vector{Float64}, l), high=convert(Vector{Float64}, h))

    # Attempts verification of condition.
    problem = Problem(r.network, input, r.output)
    result = solve(r.solver, problem)
    if result.status == :holds
        cell.data.member = true
        cell.data.proven = true
        return false
    end

    if r isa BinaryRefinery
        # Attempts verification of complement.
        problem = Problem(r.network, input, r.coutput)
        result = solve(r.solver, problem)
        if result.status == :holds
            cell.data.member = false
            cell.data.proven = true
            return false
        end
    end

    return true
end

"""
Generate data for newly-created cell.
"""
function child_data(c::Cell, idx::Tuple)
    return CellStatus(depth = c.data.depth + 1, hash = hash((c.data.hash, idx)))
end

"""
    refine!(cell::Cell, r::AbstractRefinery)

Recursively build miniminal k-d tree that defines volumes proven to satisfy given condition
and its complement. This version is single-process.
"""
function refine!(cell::Cell, r::AbstractRefinery)
    add_todo()
    yield()
    if needs_refinement(cell, r)
        split!(cell, child_data)
        for c in children(cell)
           refine!(c, r)
        end
        merge!(cell)
    end
    add_done()
    yield()
end

"""
Merge children of cell if all are proven and belong to the same set, or if all are unproven
and attached (not awaiting computation by another process). If `recursive`, all descendents
are merged.
"""
function merge!(cell::Cell; recursive::Bool=false)
    if !isleaf(cell)
        cs = children(cell)
        provens = [recursive ? merge!(c; recursive=true) : c.data.proven for c in cs]
        members = (c -> c.data.member).(cs)
        detachs = (c -> c.data.detached).(cs)
        if all(provens) && reduce(==, members) || !any(provens) && !any(detachs) && all(isleaf.(cs))
            cell.data.member = members[1]
            cell.data.proven = provens[1]
            cell.children = nothing
        end
    end
    return cell.data.proven
end

"""
Verify that tree is well-formed after recombination; i.e., no cells are marked as detached
and all children recognize the correct parent.
"""
function verify(cell::Cell)
    for c in allcells(cell)
        if c.data.detached
            return false
        end
        if !isleaf(c)
            if !all((x->x.parent == c).(children(c)))
                return false
            end
        end
    end
    return true
end

"""
Generate root from region limits.
"""
function get_root(limits::NTuple{2, Vector{<:Real}})
    lows, highs = limits
    widths = highs - lows
    root = Cell(SVector(lows...), SVector(widths...), CellStatus())
    return root
end

"""
Perform PVV analysis, using multiple processes if available.
"""
function analyze(r::AbstractRefinery, limits::Tuple{Vector, Vector}; progress::Bool=true, multiproc::Bool=true)
    TODO_COUNTER[] = 1
    tree = get_root(limits)

    @sync begin
        # Refinement process
        @async begin
            try
                if nprocs() == 1 || !multiproc
                    refine!(tree, r)
                else
                    refine_multiprocess!(tree, r)
                end
            finally
                TODO_COUNTER[] = 0
            end
        end

        # Progress meter
        if progress
            @async begin
                p = ProgressUnknown("Cells awaiting processing:")
                while TODO_COUNTER[] > 0
                    ProgressMeter.update!(p, TODO_COUNTER[]; ignore_predictor=true)
                    sleep(0.1)
                end
                ProgressMeter.update!(p, 0)
                ProgressMeter.finish!(p)
            end
        end
    end

    try
        print_metrics(tree)
    catch
        println("Unable to calculate metrics.")
    end

    return tree
end
