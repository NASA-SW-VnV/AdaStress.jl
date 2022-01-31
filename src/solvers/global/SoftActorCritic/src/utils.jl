# ******************************************************************************************
# Notices:
#
# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.
#
# Disclaimers
#
# No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF ANY KIND,
# EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY
# THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR FREEDOM FROM INFRINGEMENT, ANY
# WARRANTY THAT THE SUBJECT SOFTWARE WILL BE ERROR FREE, OR ANY WARRANTY THAT DOCUMENTATION,
# IF PROVIDED, WILL CONFORM TO THE SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT, IN ANY MANNER,
# CONSTITUTE AN ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS,
# RESULTING DESIGNS, HARDWARE, SOFTWARE PRODUCTS OR ANY OTHER APPLICATIONS RESULTING FROM
# USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY DISCLAIMS ALL WARRANTIES AND
# LIABILITIES REGARDING THIRD-PARTY SOFTWARE, IF PRESENT IN THE ORIGINAL SOFTWARE, AND
# DISTRIBUTES IT "AS IS."
#
# Waiver and Indemnity:  RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE UNITED
# STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT.
# IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS IN ANY LIABILITIES, DEMANDS, DAMAGES,
# EXPENSES OR LOSSES ARISING FROM SUCH USE, INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON,
# OR RESULTING FROM, RECIPIENT'S USE OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND
# HOLD HARMLESS THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL
# AS ANY PRIOR RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR ANY
# SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS AGREEMENT.
# ******************************************************************************************

"""
Convert DateTime to valid cross-platform filename.
"""
function dt_to_fn(dt::DateTime)
    dt = round(dt, Dates.Second)
    str = replace("$dt", ":" => "-")
    return "saved_" * str * ".bson"
end

"""
Convert filename to corresponding unix time (or NaN)
"""
function fn_to_t(fn::String)
    str = replace(fn, r"saved_(.*T)(.*)-(.*)-(.*).bson" => s"\1\2:\3:\4")
    return try datetime2unix(DateTime(str)) catch; NaN end
end

"""
Save model to specified directory, with optional maximum number of saves.
"""
function checkpoint(model, save_dir::String, max_saved::Int)

    # Delete earliest save if maximum number is exceeded
    if max_saved > 0
        files = readdir(save_dir)
        times = [fn_to_t(file) for file in files]
        if !isempty(times) && sum(@. !isnan(times)) >= max_saved
            i_min = argmin((x -> isnan(x) ? Inf : x).(times))
            rm(joinpath(save_dir, files[i_min]))
        end
    end

    # Save AC agent
    filename = joinpath(save_dir, dt_to_fn(now()))
	@save filename model
end

"""
Generate values to display/save from display tuples.
"""
function gen_showvalues(epoch::Int64, disp_tups::Vector{<:Tuple})
    () -> [(:epoch, epoch), ((sym, isempty(hist) ? NaN : hist[end]) for (sym, hist) in disp_tups)...]
end

"""
Initialize display tuples.
"""
function initialize(displays::Vector{<:Tuple})
    [(:score, []), (:stdev, []), ((sym, []) for (sym, _) in displays)...]
end

"""
Update display tuples.
"""
function update!(disp_tups::Vector{<:Tuple}, disp_vals::Vector{<:Real})
    for ((_, hist), val) in zip(disp_tups, disp_vals)
        push!(hist, val)
    end
end

"""
Set GPU status based on device availability and user preference.
"""
function set_gpu_status(use_gpu::Bool)
    WITH_GPU[] = has_cuda_gpu() && use_gpu
end

"""
Send data to device.
"""
dev(x) = WITH_GPU[] ? gpu(x) : x

"""
Recursively transfer structure to CPU in-place.
"""
function to_cpu!(x::Any, level::Int64=2)
	level < 1 && return
    if x isa Vector
        to_cpu!.(x, level)
    end
    for f in fieldnames(typeof(x))
        xf = getfield(x, f)
        level == 1 ? setfield!(x, f, cpu(xf)) : to_cpu!(xf, level - 1)
    end
end

"""
Recursively transfer copy of structure to CPU.
"""
function to_cpu(x::Any, args...)
    y = deepcopy(x)
    to_cpu!(y, args...)
    return y
end
