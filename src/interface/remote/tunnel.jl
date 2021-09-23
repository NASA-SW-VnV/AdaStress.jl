
const TUNNEL_SOCKET = Sys.iswindows() ? "" : joinpath(homedir(), ".ssh", "adastress.socket")

"""
Opens SSH tunnel to/from remote server.
"""
function open_tunnel(obj::Union{ASTClient, ASTServer}, remote::String, remote_port::Int64)

    (p1, p2, F) = obj isa ASTClient ? (obj.port, remote_port, "L") : (remote_port, obj.port, "R")

    if Sys.iswindows()
        msg = """
        Port forwarding cannot be established from within Julia on Windows.

        Run command `ssh -$F $p1:localhost:$p2 $remote` in a seperate terminal
        window and leave open, then rerun `connect!` with keyword `external` set to true.
        """
        throw(ErrorException(msg))
    end

    run(`ssh -fN -M -S $TUNNEL_SOCKET -$F $p1:localhost:$p2 $remote`)
    obj.tunnel = true
    return
end

"""
Closes SSH tunnel to/from remote server.
"""
function close_tunnel()
    if Sys.iswindows()
        @warn """
        \nPort forwarding cannot be stopped from within Julia on Windows.
        Close seperate terminal window to fully terminate connection.
        """
    else
        run(`ssh -S $TUNNEL_SOCKET -O exit ""`)
    end
    return
end
