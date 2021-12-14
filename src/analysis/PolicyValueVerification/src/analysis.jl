
"""
    num_leaves(cell::Cell)

Count number of leaves below cell.
"""
num_leaves(cell::Cell) = sum(1 for _ in allleaves(cell))

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
    print_metrics(cell::Cell)

Print metrics about proof coverage and tree size.
"""
function print_metrics(cell::Cell)
    vtrue = coverage(cell, true)
    vfalse = coverage(cell, false)
    nleaves = num_leaves(cell)
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

    println("Vol. proportion proven true  : $(vtrue)")
    println("Vol. proportion proven false : $(vfalse)")
    println("Vol. proportion unproven     : $(1 - vtrue - vfalse)")
    println("Total size of spanning tree  : $(nleaves) leaves")
    println("Total size of tree in memory : $val $(pre)B")
end
