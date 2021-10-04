
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
