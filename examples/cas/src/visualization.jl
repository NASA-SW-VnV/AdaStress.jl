
RA_COLOR = Dict(:cw => :green, :ccw => :blue, :none => :gray)
RA_SHAPE = Dict(:asc => :utriangle, :desc => :dtriangle, :none => :circle)
FORMAT = (grid=true, gridalpha=0.2, minorgrid=true, minorgridalpha=0.1, legend=:none)

function parse_connect(log, connect)
    if connect == :none
        []
    elseif connect == :all
        subsets(1:log["n"], 2)
    elseif connect isa Tuple{Z, Z}
        [connect]
    elseif connect isa Vector{Tuple{Z, Z}}
        connect
    else
        throw(ArgumentError("keyword `connect` does not support value $connect."))
    end
end

function add_horizontal(log::Dict{String, Any}; connect=:all)
	p = plot(; FORMAT..., title="Horizontal Position")
	ss = []
	for i in 1:log["n"]
		s = reduce(vcat, s' for s in log["ac_$i"])
		push!(ss, s)
		plot!(p, s[:,1], s[:,2]; color=i, aspect_ratio=:equal, FORMAT...)
		scatter!(p, [s[1,1]], [s[1,2]]; ms=3, mc=:black, shape=:star, FORMAT...)

		# advisories
		ras = log["cas_$i"]
		for (t, ra) in enumerate(ras)
			if any(ra .!= :none)
				color = RA_COLOR[ra[1]]
				shape = RA_SHAPE[ra[2]]
				scatter!(p, s[t,1:1], s[t,2:2]; ms=3, msw=0, mc=color, shape=shape, FORMAT...)
			end
		end
	end

    # add inter-aircraft connections
    for (i, j) in parse_connect(log, connect)
	    for t in 1:size(ss[1], 1)
            xs = (s -> s[t,1]).(ss[[i,j]])
            ys = (s -> s[t,2]).(ss[[i,j]])
		    plot!(p, xs, ys; lc=:black, ls=:dot, lw=0.5, ma=0.5, FORMAT...)
        end
	end

	return p
end

function add_vertical(log::Dict{String, Any})
	p = plot(title="Vertical Position")
    ts = log["t"]

	for i in 1:log["n"]
		s = reduce(vcat, s' for s in log["ac_$i"])
		plot!(p, ts, s[:,3]; color=i, FORMAT...)

		# advisories
		ras = log["cas_$i"]
		for (t, ra) in enumerate(ras)
			if any(ra .!= :none)
				color = RA_COLOR[ra[1]]
				shape = RA_SHAPE[ra[2]]
				scatter!(p, ts[t:t], s[t,3:3]; ms=3, msw=0, mc=color, shape=shape, FORMAT...)
			end
		end
	end
	return p
end

function add_separation(log::Dict{String, Any})
	return plot(log["t"], log["d"]; title="Separation", FORMAT...)
end

function add_velocity(log::Dict{String, Any})
	p = plot(title="Ground Velocity")
	for i in 1:log["n"]
		s = reduce(vcat, s' for s in log["ac_$i"])
		plot!(p, log["t"], sqrt.(s[:,4].^2 + s[:,5].^2); FORMAT...)
	end
	return p
end

function visualize(log::Dict{String, Any}; filename::String="", kwargs...)
    @assert !isempty(log)

	ps = [
        add_horizontal(log; kwargs...),
        add_vertical(log),
        add_separation(log),
        add_velocity(log)
    ]

	p = plot(ps..., layout=(2,2), size=(900,500))
    isempty(filename) || savefig(p, filename)
	return p
end

visualize(sim::Simulator; kwargs...) = visualize(sim.log; kwargs...)
