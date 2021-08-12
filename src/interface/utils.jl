"""
Allows global random seed to be set in limited scope without affecting larger program.
"""
macro fix(seed, ex)
    quote
        rng = Random.default_rng()
        rng_save = deepcopy(rng)
        Random.seed!(rng, $(esc(seed)))
        output = $(esc(ex))
        rng = rng_save
        output
    end
end
