
"""
Contains cell information pertaining to proof and multiprocessing status.
"""
Base.@kwdef mutable struct CellStatus
    depth::Int64 = 0
    detached::Bool = false
    hash::UInt64 = hash(0)
    member::Bool = false
    proven::Bool = false
end

"""
Defines parameters of binary refinement process.
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
Defines parameters of interval refinement process.
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
Determines whether condition can be proven true or false over entire cell
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
    input = Hyperrectangle(low=convert(Vector{Float64}, l),
                           high=convert(Vector{Float64}, h))

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
Generates data for newly-created cell.
"""
function child_data(c::Cell, idx::Tuple)
    return CellStatus(depth = c.data.depth + 1, hash = hash((c.data.hash, idx)))
end

"""
Recursively builds miniminal k-d tree that defines volumes proven to satisfy
given condition and its complement. This version is single-process.
"""
function refine!(cell::Cell, r::AbstractRefinery)
    if needs_refinement(cell, r)
        split!(cell, child_data)
        for c in children(cell)
           refine!(c, r)
        end
        merge!(cell)
    end
    return nothing
end

"""
Merges children of cell if all are proven and belong to the same set, or if
all are unproven and attached (not awaiting computation by another process).
If recursive = true, all descendents are merged.
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
Verify that tree is well-formed after recombination; i.e., no cells
are marked as detached and all children recognize the correct parent.
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
Generates root from region limits.
"""
function get_root(limits::NTuple{2, Vector{Float64}})
    lows, highs = limits
    widths = highs - lows
    root = Cell(SVector(lows...), SVector(widths...), CellStatus())
    return root
end
