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

#=
These functions automatically manage SSH tunneling, allowing a remote AdaStress architecture
to be established over secure channels. They do not need to be invoked directly, since they
are used by the `connect!` and `disconnect!` functions when the appropriate keyword
arguments are set.

The Linux/Apple version uses SSH multiplexing to maintain a handle on the SSH session, while
the Windows version stores the raw SSH process id.
=#

const TUNNEL_SOCK = Sys.iswindows() ? "" : joinpath(homedir(), ".ssh", "adastress.sock")
const TUNNEL_PROC = Ref(0)

"""
Open SSH tunnel to/from remote server.
"""
function open_tunnel(obj::Union{ASTClient, ASTServer}, remote::String, remote_port::Int64)
    (p1, p2, F) = obj isa ASTClient ? (obj.port, remote_port, "L") : (remote_port, obj.port, "R")

    if Sys.iswindows()
        ssh_args = "-$F $p1:localhost:$p2 $remote"
        pwsh_cmd = "(Start-Process ssh -Passthru -ArgumentList \"$ssh_args\").id"
        pipe = pipeline(`cmd /c powershell $pwsh_cmd`)
        out = read(pipe, String) # capturing the ssh pid
        TUNNEL_PROC[] = parse(Int64, match(r"\d*", out).match)
        @info """
        Enter credentials in pop-up terminal window, leave window open, and return to Julia.
        Press enter to continue...
        """
        readline()
    else
        ssh_args = `-fN -M -S $TUNNEL_SOCK -$F $p1:localhost:$p2 $remote`
        Base.run(`ssh $ssh_args`)
    end

    @info "Tunnel is open."
    obj.tunnel = true
    return
end

"""
Close SSH tunnel to/from remote server.
"""
function close_tunnel()
    if Sys.iswindows()
        pwsh_cmd = "kill $(TUNNEL_PROC[])"
        Base.run(`cmd /c powershell $pwsh_cmd`)
        TUNNEL_PROC[] = 0
    else
        ssh_args = `-S $TUNNEL_SOCK -O exit ""`
        Base.run(`ssh $ssh_args`)
    end
    @info "Tunnel is closed."
    return
end
