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
Convert cell to `Shape` type.
"""
shape(cell::Cell) = Shape(Tuple.(collect(vertices(cell).data))[[1,2,4,3]])

"""
Return x- and y-coordinates of cell vertices.
"""
function vs(cell::Cell)
    v = hcat(collect(vertices(cell))...)
    vx = v[1,[1,2,4,3,1]]
    vy = v[2,[1,2,4,3,1]]
    return vx, vy
end

FILL_OPTIONS = [:none, :proof, :pid]

"""
    visualize!([p], root::Cell; fill::Symbol=:none, tol::Float64=0.01)

Visualize edges and/or shapes of cells. Optional first argument is a pre-existing plot.
"""
function visualize!(p, root::Cell; fill::Symbol=:none, tol::Float64=0.01)
    # plot parameters
    @assert fill in FILL_OPTIONS
    format = (alpha=1.0, lw=0.0, label=:none)

    # gray background
    fill == :none || plot!(p, shape(root); color=:gray, format...)

    # iterate through leaves
    pr = Progress(num_leaves(root), 1)
    for leaf in allleaves(root)
        w = leaf.boundary.widths
        if minimum(w) >= tol
            if fill == :none
                plot!(p, vs(leaf)..., color=:black, lw=0.5, label=:none)
            elseif fill == :proof
                if leaf.data.proven
                    color = leaf.data.member ? :green : :red
                    plot!(p, shape(leaf); color=color, format...)
                end
            elseif fill == :pid
                plot!(p, shape(leaf); color=leaf.data.pid, format...)
            end
        end
        next!(pr)
    end
    return p
end

visualize(p, root::Cell; kwargs...) = visualize!(deepcopy(p), root; kwargs...)
visualize(root::Cell; kwargs...) = visualize!(plot(), root; kwargs...)

"""
    visualize(nnet::Network, limits::NTuple{2, Vector{Float64}})

Visualize network cross-section.
"""
function visualize(nnet::Network, limits::NTuple{2, Vector{Float64}})
    if size(nnet.layers[1].weights, 2) != 2
        error("Number of free variables must be 2.")
    end

    n = 100
    f = (nnet, x, y) -> compute_output(nnet, [x, y])[]
    xlims = getindex.(limits, 1)
    ylims = getindex.(limits, 2)
    xs = collect(range(xlims..., length=n))
    ys = collect(range(ylims..., length=n))
    zs = f.(Ref(nnet), xs, ys')
    p = heatmap(xs, ys, zs'; aspect_ratio=:equal, xlims=xlims, ylims=ylims)
    return p
end
