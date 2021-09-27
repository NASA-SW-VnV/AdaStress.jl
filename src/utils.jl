#=
Contains implementation of submodule manager. Submodules are optional modules with heavy
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
Associates package directory to main package via submodule table instead of code loading.
Equivalent of `include` for submodules / optional dependencies.
"""
function exclude(file::String)
    fprev = stacktrace()[2].file
    path = joinpath(dirname(String(fprev)), file)
    name = basename(file) # directory name must match corresponding module name
    SUBMODULES[name] = path
    return
end

"""
Lists all associated submodules.
"""
submodules() = collect(keys(SUBMODULES))

"""
Lists enabled submodules.
"""
function enabled()
    curr = Pkg.project().path
    @suppress Pkg.activate(ENV_DIR)
    deps = filter(p -> p[2].is_direct_dep, Pkg.dependencies())
    names = (d -> d.name).(values(deps))
    @suppress Pkg.activate(curr)
    return filter(d -> d != PKG_NAME && d in keys(SUBMODULES), names)
end

"""
Lists unregistered dependencies of submodule.
"""
function unregistered_deps(submodule::String)
    path = SUBMODULES[submodule]
    toml = TOML.parsefile(joinpath(path, "Project.toml"))
    return "unreg" in keys(toml) ? toml["unreg"] : Dict{String, Any}()
end

"""
Enables submodule. Can accept multiple strings as vector or multiple arguments.
With zero arguments defaults to all associated submodules. Takes effect immediately.
"""
function enable(submodule::String; verbose::Bool=true)
    curr = Pkg.project().path
    try
        @suppress begin
            Pkg.activate(ENV_DIR)
            Pkg.develop(path=PKG_PATH) # updates AdaStress if necessary

            dev_pkgs = readdir(DEV_DIR)
            for (dep, url) in unregistered_deps(submodule)
                dep in dev_pkgs ? Pkg.develop(dep) : Pkg.develop(url=url)
            end

            Pkg.develop(path=SUBMODULES[submodule])
            @eval using $(Symbol(submodule))
        end
        verbose && @info "Enabled submodule $submodule."
    catch e
        verbose && @error "Unable to enable submodule $submodule."

        Pkg.rm(submodule)
        for (dep, _) in unregistered_deps(submodule)
            Pkg.rm(dep)
        end

        throw(e)
    finally
        @suppress Pkg.activate(curr)
    end
    return
end

enable(submodules::Vector; verbose::Bool=true) = (enable.(submodules; verbose=verbose); nothing)
enable(args...; verbose::Bool=true) = enable(collect(args); verbose=verbose)
enable(; verbose::Bool=true) = enable(collect(keys(SUBMODULES)); verbose=verbose)

"""
Disables submodule. Can accept multiple strings as vector or multiple arguments.
With zero arguments defaults to all enabled submodules. Takes effect after Julia restart.
"""
function disable(submodule::String; verbose::Bool=true)
    curr = Pkg.project().path
    try
        @suppress begin
            Pkg.activate(ENV_DIR)
            Pkg.rm(submodule)

            for (dep, _) in unregistered_deps(submodule)
                # NOTE: will cause problems if multiple enabled submodules use the same
                #       unregistered dependencies, but this is unlikely, given the ideally
                #       rare usage of such dependencies.
                Pkg.rm(dep)
            end
        end
        verbose && @info "Disabled submodule $submodule."
    catch e
        verbose && @error "Unable to disable submodule $submodule."
        throw(e)
    finally
        @suppress Pkg.activate(curr)
    end
    return
end

disable(submodules::Vector; verbose::Bool=true) = (disable.(submodules; verbose=verbose); nothing)
disable(args...; verbose::Bool=true) = disable(collect(args); verbose=verbose)
disable(; verbose::Bool=true) = disable(enabled(); verbose=verbose)

"""
Loads enabled submodules (required after Julia restart). Takes effect immediately.
"""
function load(; verbose::Bool=true)
    curr = Pkg.project().path
    try
        @suppress Pkg.activate(ENV_DIR)
        for submodule in enabled()
            @suppress @eval using $(Symbol(submodule))
            verbose && @info "Loaded submodule $submodule."
        end
    catch e
        verbose && @error "Unable to load enabled submodules."
        throw(e)
    finally
        @suppress Pkg.activate(curr)
    end
    return
end

"""
Forcibly removes temporary environment, purging all enabled submodules. Only required if
submodule manager becomes corrupted and `disable()` cannot restore functionality. Takes effect after
Julia restart.
"""
function clean(; verbose::Bool=true)
    rm.(joinpath.(ENV_DIR, readdir(ENV_DIR)))
    verbose && @info "Cleaned submodule environment."
end

"""
Invokes any submodule-related initializations. Called by top-level __init__() function.
"""
function init_submodules()
    # scratchspace for intermediate projects
    global ENV_DIR = @get_scratch!("env")
end
