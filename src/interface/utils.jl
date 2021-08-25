"""
Temporary storage for any RNG of the same type as the global RNG.
"""
RNG_TEMP = deepcopy(Random.default_rng())

"""
Partial function application transforming f(s,s') to f(s)(s').
Constructs anonymous function out of expression with input mdp::ASTMDP.
"""
macro defer(expr)
    esc(:(mdp -> $expr))
end
