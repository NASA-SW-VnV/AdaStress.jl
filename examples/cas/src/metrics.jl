"""
Cache of expensive and potentially repeatedly-referenced computations.
"""
Base.@kwdef mutable struct Metrics
    d::Dict{String, Any} = Dict{String, Any}()
end

function initialize(m::Metrics, acs::Vector{Aircraft}, nmac::Tuple{R,R})
    m.d["d_min"] = Inf
    update(m, acs, nmac)
end

function update(m::Metrics, acs::Vector{Aircraft}, nmac::Tuple{R,R})
	pairs = subsets(acs, 2)
	m.d["d"] = minimum(norm(separation(pair...)) for pair in pairs) #TODO: was separation(pair...; scale=nmac)
    m.d["d_min"] = min(m.d["d_min"], m.d["d"])
	m.d["is_nmac"] = any(all(separation(pair...) .<= nmac) for pair in pairs)
end
