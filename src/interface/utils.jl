
RNG_TEMP = deepcopy(Random.default_rng()) # temporary storage for RNG of same type as global

"""
Conditional function. Applies function if applicable, otherwise returns value. Useful for
conditional partial function application.
"""
struct Functoid{T<:Any}
    val::T
end

(f::Functoid{<:Any})(args...; kwargs...) = f.val
(f::Functoid{<:Function})(args...; kwargs...) = f.val(args...; kwargs...)
