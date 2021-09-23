#TODO: robustness, isopen checks, reconnection

FLAGS = [:reset!, :actions, :observe, :act!, :terminated]
FMAP = Bijection(Dict(UInt8(i) => f for (i, f) in enumerate(FLAGS))) # maps flags <-> bytes

"""
Solver-side client. Interacts with server via TCP.
"""
Base.@kwdef mutable struct ASTClient{S<:State, A<:Action} <: AbstractASTMDP{S, A}
    ip::IPAddr                      = ip"::1"   # ip of server
    port::Int64                     = 1812      # server port
    conn::Union{TCPSocket, Nothing} = nothing
    tunnel::Bool                    = false
    verbose::Bool                   = false
end

"""
Performs virtual function call on remote MDP. Sends request to server with
provided function and arguments and receives return value. Blocks until complete.
"""
function call(client::ASTClient, f::Function, args...)
    flag = Symbol(f)
    request = Dict(:f => FMAP(flag), :a => args)
    bson(client.conn, request)
    client.verbose && @info "Sending request to server:" request
    response = BSON.load(client.conn)
    client.verbose && @info "Received response from server:" response
    return response[:r]
end

reset!(client::ASTClient) = call(client, reset!)

actions(client::ASTClient) = call(client, actions)

observe(client::ASTClient) = call(client, observe)

act!(client::ASTClient, action) = call(client, act!, action)

terminated(client::ASTClient) = call(client, terminated)

"""
Requests ping from ASTServer. Returns round-trip time in seconds.
"""
function ping(client::ASTClient)
    request = Dict(:f => 0x0)
    t = () -> datetime2unix(now())
    t1 = t()
    bson(client.conn, request)
    BSON.load(client.conn) #TODO: include info payload?
    return t() - t1
end

"""
Connects client to server, optionally through SSH tunnel.
Optional argument `remote` should be of the form `user@machine`.
"""
function connect!(client::ASTClient; remote::String="", remote_port::Int64=1812)
    disconnect!(client)
    !isempty(remote) && open_tunnel(client, remote, remote_port)
    client.conn = connect(client.ip, client.port)
    t_ms = Int64(1000 * round(ping(client); digits=3))
    @info "ASTServer responded in $t_ms milliseconds."
    return
end

"""
Disconnects client from server.
"""
function disconnect!(client::ASTClient)
    if client.conn !== nothing
        close(client.conn)
        client.conn = nothing
    end
    client.tunnel && close_tunnel()
    client.tunnel = false
    return
end
