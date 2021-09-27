"""
Temporary storage for any RNG of the same type as the global RNG.
"""
RNG_TEMP = deepcopy(Random.default_rng())

"""
Conditional function. Applies function if applicable, otherwise returns value. Useful for
conditional partial function application.
"""
struct Functoid{T<:Any}
    val::T
end

(f::Functoid{<:Any})(args...; kwargs...) = f.val
(f::Functoid{<:Function})(args...; kwargs...) = f.val(args...; kwargs...)
