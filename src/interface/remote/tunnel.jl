
const TUNNEL_SOCKET = Sys.iswindows() ? "" : joinpath(homedir(), ".ssh", "adastress.socket")
const TUNNEL_PROC = Ref(0)

"""
Opens SSH tunnel to/from remote server. Called during `connect!` if keywords are set.
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
        ssh_args = `-fN -M -S $TUNNEL_SOCKET -$F $p1:localhost:$p2 $remote`
        Base.run(`ssh $ssh_args`)
    end

    @info "Tunnel is open."
    obj.tunnel = true
    return
end

"""
Closes SSH tunnel to/from remote server. Called during `disconnect!` if keywords are set.
"""
function close_tunnel()
    if Sys.iswindows()
        pwsh_cmd = "kill $(TUNNEL_PROC[])"
        Base.run(`cmd /c powershell $pwsh_cmd`)
        TUNNEL_PROC[] = 0
    else
        ssh_args = `-S $TUNNEL_SOCKET -O exit ""`
        Base.run(`ssh $ssh_args`)
    end
    @info "Tunnel is closed."
    return
end
