#TODO: add security, add robustness, add isopen checks / reconnection, compress commands into flags (choose simpler serialization?)
#TODO: maybe to preserve privacy division, actions(.) should automatically cause client to return rand(actions(.)),
# then server creates thin sampleable wrapper around it.

# Current design:
# Server contains sim (external to NASA)
# Client interacts with solvers, etc. (internal to NASA)

mutable struct ASTClient <: CommonRLInterface.AbstractEnv
    ip::IPAddr          # ip of server
    port::Int64         # server port
    conn::TCPSocket
end

function ASTClient(ip::IPAddr, port::Int64)
    conn = connect(ip, port)
    return ASTClient(ip, port, conn)
end

function CommonRLInterface.reset!(client::ASTClient)
    bson(client.conn, Dict(:f => :reset))
    ack = BSON.load(client.conn)
    return ack[:z]
end

#TODO: maybe cache this locally? not safe if environment changes.
#TODO: see idea in header
function CommonRLInterface.actions(client::ASTClient)
    bson(client.conn, Dict(:f => :actions))
    env = BSON.load(client.conn)
    return Environment(env[:z])
end

function CommonRLInterface.observe(client::ASTClient)
    bson(client.conn, Dict(:f => :observe))
    obs = BSON.load(client.conn)
    return obs[:z]
end

function CommonRLInterface.act!(client::ASTClient, action::Vector{<:Real})
	bson(client.conn, Dict(:f => :act, :a => action))
    r = BSON.load(client.conn)
	return r[:z]
end

function CommonRLInterface.terminated(client::ASTClient)
    bson(client.conn, Dict(:f => :terminated))
    d = BSON.load(client.conn)
    return d[:z]
end

mutable struct ASTServer <: CommonRLInterface.AbstractEnv
    ip::IPAddr
    port::Int64
    server::Sockets.TCPServer
    conn::TCPSocket
    mdp::ASTMDP
end

#TODO: break this into multiple functions so server can be created independent of connectivity
function ASTServer(mdp::ASTMDP, ip::IPAddr, port::Int64)
    server = listen(ip, port)
    conn = accept(server)
    return ASTServer(ip, port, server, conn, mdp)
end

#TODO: replace with bijection object (maybe from Bijection.jl)
# which maps CommonRLInterface functions to UInt8
function req2func(sym::Symbol)
    if sym == :reset
        return CommonRLInterface.reset!
    elseif sym == :actions
        return CommonRLInterface.actions
    elseif sym == :observe
        return CommonRLInterface.observe
    elseif sym == :act
        return CommonRLInterface.act!
    elseif sym == :terminated
        return CommonRLInterface.terminated
    else
        error("Invalid request.")
    end
end

function run(server::ASTServer)
    @async while true
        request = BSON.load(server.conn)
        f = req2func(request[:f])
        @info "Received request from client" f
        ret = haskey(request, :a) ? f(server.mdp, request[:a]) : f(server.mdp) #TODO: improve
        bson(server.conn, Dict(:z => ret))
        @info "Sent response to client"
    end
end
