
const PKG_PATH = @__DIR__
const DEFAULT_OUTDIR = joinpath(PKG_PATH, "../output")
mkpath(DEFAULT_OUTDIR)

const R = Float64
const Z = Int64
const G = 32.17 # gravitational acceleration [ft/s^2]

wrap_to_pi(x::R) =  x - (2π * floor((x + π) / (2π)))

"""
Fill array from expression evaluation.
"""
macro array(n, body)
    :([$(esc(body)) for _ in 1:$(esc(n))])
end
