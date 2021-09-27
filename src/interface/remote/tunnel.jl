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
Opens SSH tunnel to/from remote server.
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
        Please enter credentials, then leave terminal window open and return to Julia.
        Press any key to continue...
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
Closes SSH tunnel to/from remote server.
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
