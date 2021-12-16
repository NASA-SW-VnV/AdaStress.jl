
"""
    num_leaves(cell::Cell)

Count number of leaves below cell.
"""
num_leaves(cell::Cell) = sum(1 for _ in allleaves(cell))

"""
    max_depth(cell::Cell)

Determine maximum depth below cell.
"""
max_depth(cell::Cell) = isleaf(cell) ? 1 : 1 + maximum(max_depth.(children(cell)))

"""
Total volume of cell proven to satisfy condition or complement.
"""
function proven_volume(cell::Cell, member::Bool=true)
    vol = 0.0
    for l in allleaves(cell)
        if l.data.proven && l.data.member == member
            vol += prod(l.boundary.widths)
        end
    end
    return vol
end

"""
    coverage(cell::Cell, member::Union{Bool,Nothing}=nothing)

Proportion of cell volume proven to satisfy condition, complement, or either.
"""
function coverage(cell::Cell, member::Union{Bool,Nothing}=nothing)
    if member === nothing
        vol = proven_volume(cell, true) + proven_volume(cell, false)
    else
        vol = proven_volume(cell, member)
    end

    return vol / prod(cell.boundary.widths)
end

"""
Compute balance of tree over processors, scaled to [0, 1].
"""
function balance_score(cell::Cell)
    nworkers() == 0 && return NaN

    counts = [0 for _ in workers()]
    for c in allcells(cell)
        if c.data.pid > 0
            counts[c.data.pid - 1] += 1
        end
    end

    dist = counts / sum(counts)
    n = length(dist)
    d = sqrt(sum(dist.^2))
    return sqrt(n) * (1 - d) / (sqrt(n) - 1)
end

"""
Compute size of tree in memory and return formatted variables.
"""
function memory_size(cell::Cell)
    sz = Base.summarysize(cell)
    prefixes = ["G", "M", "k", ""]
    sizes = [1_000_000_000, 1_000_000, 1_000, 1]
    val, pre = 0.0, ""

    for (s, p) in zip(sizes, prefixes)
        if sz >= s
            val = sz / s
            pre = p
            break
        end
    end
    return val, pre
end

"""
    print_metrics(cell::Cell)

Print metrics about proof coverage and tree size.
"""
function print_metrics(cell::Cell)
    vtrue = coverage(cell, true)
    vfalse = coverage(cell, false)
    nleaves = num_leaves(cell)
    val, pre = memory_size(cell)
    dep = max_depth(cell)
    bal = balance_score(cell)

    println("Vol. proportion proven true  : $(vtrue)")
    println("Vol. proportion proven false : $(vfalse)")
    println("Vol. proportion unproven     : $(1 - vtrue - vfalse)")
    println("Total size of spanning tree  : $(nleaves) leaves")
    println("Total size of tree in memory : $val $(pre)B")
    println("Maximum depth of tree        : $dep")
    println("Processor balancing score    : $bal")
end
