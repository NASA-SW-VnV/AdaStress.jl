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
    response = BSON.load(client.conn)
    return response[:r]
end

function reset!(client::ASTClient)
    client.verbose && @info "Sending request to server: `reset!`"
    call(client, reset!)
end

function actions(client::ASTClient)
    client.verbose && @info "Sending request to server: `actions`"
    call(client, actions)
end

function observe(client::ASTClient)
    client.verbose && @info "Sending request to server: `observe`"
    call(client, observe)
end

function act!(client::ASTClient, action)
    client.verbose && @info "Sending request to server: `act!`"
    call(client, act!, action)
end

function terminated(client::ASTClient)
    client.verbose && @info "Sending request to server: `terminated`"
    call(client, terminated)
end

"""
Connects client to server, optionally through SSH tunnel.
Optional argument `remote` should be of the form `user@machine`.
"""
function connect!(client::ASTClient; remote::String, remote_port::Int64=1812, external::Bool=false)
    disconnect!(client)
    !isempty(remote) && !external && open_tunnel(client, remote, remote_port)
    client.tunnel = external ? true : server.tunnel

    dt = @elapsed client.conn = connect(client.ip, client.port)
    @info "Connected to AST server in $dt seconds." client.conn
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
    return
end
