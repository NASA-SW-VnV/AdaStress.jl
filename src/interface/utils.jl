"""
Temporary storage for any RNG of the same type as the global RNG.
"""
RNG_TEMP = deepcopy(Random.default_rng())

"""
Partial function application transforming f(s,s') to f(s)(s').
Constructs anonymous function out of expression with input mdp′::ASTMDP, where
mdp′ represents the post-step mdp.
"""
macro defer(expr)
    esc(:(mdp′::ASTMDP -> $expr))
end
