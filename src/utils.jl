#=
Implementation of submodule manager. Submodules are optional modules with heavy
dependencies. See documentation for usage and implementation details.
=#

using Pkg
using Scratch
using Suppressor
using TOML

ENV_DIR = ""                                                # directory of submodule environment
const DEV_DIR = mkpath(joinpath(first(DEPOT_PATH), "dev"))  # directory of packages added via `dev`
const PKG_NAME = "$(@__MODULE__)"                           # top-level package name (AdaStress)
const PKG_PATH = dirname(dirname(pathof(@__MODULE__)))      # top-level package directory
const SUBMODULES = Dict{String, String}()                   # submodule table

"""
Suppresses output if verbosity condition is not met.
"""
macro verboseif(v, expr)
    :($(esc(v)) ? $(esc(expr)) : @suppress $(esc(expr)))
end

"""
    exclude(dir::String)

Associate package directory to main package via submodule table instead of direct code
loading. Equivalent of `include` for submodules / optional dependencies.
"""
function exclude(dir::String)
    fprev = stacktrace()[2].file
    path = joinpath(dirname(String(fprev)), dir)
    name = basename(dir) # directory name must match corresponding module name
    SUBMODULES[name] = path
    return
end

"""
    submodules()

List all available submodules.
"""
submodules() = collect(keys(SUBMODULES))

"""
    enabled()

List enabled submodules.
"""
function enabled(; verbose::Int64=1)
    curr = Pkg.project().path
    @verboseif (verbose >= 2) Pkg.activate(ENV_DIR)
    deps = filter(p -> p[2].is_direct_dep, Pkg.dependencies())
    names = (d -> d.name).(values(deps))
    @verboseif (verbose >= 2) Pkg.activate(curr)
    return filter(d -> d != PKG_NAME && d in keys(SUBMODULES), names)
end

"""
List unregistered dependencies of submodule.
"""
function unregistered_deps(submodule::String)
    path = SUBMODULES[submodule]
    toml = TOML.parsefile(joinpath(path, "Project.toml"))
    return "unreg" in keys(toml) ? toml["unreg"] : Dict{String, Any}()
end

"""
    enable(submodule)

Enable submodule(s). Accepts string or vector of strings. With zero arguments defaults to
all associated submodules. Takes effect immediately.
"""
function enable(submodules::Vector{String}; verbose::Int64=1)
    curr = Pkg.project().path
    pkgs = [PackageSpec(path=PKG_PATH)] # update AdaStress if necessary
    dev_pkgs = readdir(DEV_DIR)
    for submodule in submodules
        for (dep, url) in unregistered_deps(submodule)
            push!(pkgs, dep in dev_pkgs ? PackageSpec(name=dep) : PackageSpec(url=url))
        end
        push!(pkgs, PackageSpec(path=SUBMODULES[submodule]))
    end

    try
        @verboseif (verbose >= 2) begin
            Pkg.activate(ENV_DIR)
            Pkg.develop(pkgs)
            for submodule in submodules
                @eval using $(Symbol(submodule))
            end
        end
        @verboseif (verbose >= 1) foreach(s -> @info("Enabled submodule $s."), submodules)
    catch e
        @verboseif (verbose >= 1) @error "Cannot enable submodule(s)."
        @verboseif (verbose >= 2) foreach(pkg -> try Pkg.rm(pkg) catch end, pkgs[2:end])
        throw(e)
    finally
        @verboseif (verbose >= 2) Pkg.activate(curr)
    end
    return
end

enable(submodule::String; kwargs...) = enable([submodule]; kwargs...)
enable(; kwargs...) = enable(submodules(); kwargs...)

"""
    disable(submodule)

Disable submodule(s). Accepts string or vector of strings. With zero arguments defaults to
all enabled submodules. Takes effect after Julia restart.
"""
function disable(submodules::Vector{String}; verbose::Int64=1)
    curr = Pkg.project().path
    pkgs = String[]
    for submodule in submodules
        push!(pkgs, submodule)
        for (dep, _) in unregistered_deps(submodule)
            # NOTE: will cause problems if multiple enabled submodules use the same
            #       unregistered dependencies, but this is unlikely, given the ideally
            #       rare usage of such dependencies.
            push!(pkgs, dep)
        end
    end

    try
        @verboseif (verbose >= 2) begin
            Pkg.activate(ENV_DIR)
            Pkg.rm(pkgs)
        end
        @verboseif (verbose >= 1) foreach(s -> @info("Disabled submodule $s."), submodules)
    catch e
        @verboseif (verbose >= 1) @error "Cannot disable submodule(s)."
        throw(e)
    finally
        @verboseif (verbose >= 2) Pkg.activate(curr)
    end
    return
end

disable(submodule::String; kwargs...) = disable([submodule]; kwargs...)
disable(; kwargs...) = disable(enabled(; kwargs...); kwargs...)

"""
    load()

Load enabled submodules (necessary after Julia restart). Takes effect immediately.
"""
function load(; verbose::Int64=1)
    curr = Pkg.project().path
    try
        @verboseif (verbose >= 2) Pkg.activate(ENV_DIR)
        for submodule in enabled()
            @suppress @eval using $(Symbol(submodule))
            @verboseif (verbose >= 1) @info "Loaded submodule $submodule."
        end
    catch e
        @verboseif (verbose >= 1) @error "Unable to load enabled submodules."
        throw(e)
    finally
        @verboseif (verbose >= 2) Pkg.activate(curr)
    end
    return
end

"""
    clean()

Forcibly remove temporary environment, purging all enabled submodules. Only necessary if
submodule manager is corrupted and `disable` cannot restore functionality. Takes effect
after Julia restart.
"""
function clean(; verbose::Int64=1)
    rm.(joinpath.(ENV_DIR, readdir(ENV_DIR)))
    @verboseif (verbose >= 1) @info "Cleaned submodule environment."
end

"""
Invoke any submodule-related initializations. Called by top-level `__init__` function.
"""
function init_submodules()
    # scratchspace for intermediate projects
    global ENV_DIR = @get_scratch!("env")
end
