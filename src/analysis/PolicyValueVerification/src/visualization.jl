
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

"""
    visualize!([p], root::Cell; fill::Bool=false, tol::Float64=0.01)

Visualize edges and/or shapes of cells. Optional first argument is a pre-existing plot.
"""
function visualize!(p, root::Cell; fill::Bool=false, tol::Float64=0.01)
    # plot parameters
    format = (alpha=1.0, lw=0.0, label=:none)

    # gray background
    fill && plot!(p, shape(root); color=:gray, format...)

    # iterate through leaves
    pr = Progress(num_leaves(root), 1)
    for leaf in allleaves(root)
        w = leaf.boundary.widths
        if minimum(w) >= tol
            if fill
                if leaf.data.proven
                    color = leaf.data.member ? :green : :red
                    plot!(p, shape(leaf); color=color, format...)
                end
            else
                plot!(p, vs(leaf)..., color=:black, lw=0.5, label=:none)
            end
        end
        next!(pr)
    end
    return p
end

visualize(root::Cell; kwargs...) = visualize!(plot(), root; kwargs...)

"""
    visualize(network::ExtendedNetwork, cs::CrossSection, limits::NTuple{2, Vector{Float64}})

Visualize network cross-section.
"""
function visualize(network::ExtendedNetwork, cs::CrossSection, limits::NTuple{2, Vector{Float64}})
    return visualize(network, linearize(cs), limits)
end

function visualize(network::ExtendedNetwork, lcs::LinearCrossSection, limits::NTuple{2, Vector{Float64}})
    nnet = cross_section(network, lcs, limits)
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
