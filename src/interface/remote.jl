#TODO: security, robustness, isopen checks, reconnection
#TODO: create test
#TODO: ensure works with ssh

FLAGS = [:reset!, :actions, :observe, :act!, :terminated]
FMAP = Bijection(Dict(UInt8(i) => f for (i, f) in enumerate(FLAGS))) # maps flags <-> bytes

"""
Solver-side client. Interacts with server via TCP.
"""
Base.@kwdef mutable struct ASTClient <: CommonRLInterface.AbstractEnv
    ip::IPAddr                      = IPv4(0)   # ip of server
    port::Int64                     = 2000      # server port
    conn::Union{TCPSocket, Nothing} = nothing
    verbose::Bool                   = false
end

"""
Disconnects client from server.
"""
function disconnect!(client::ASTClient)
    if client.conn !== nothing
        close(client.conn)
        client.conn = nothing
    end
end

"""
Connects client with server.
"""
function connect!(client::ASTClient)
    disconnect!(client)
    dt = @elapsed client.conn = connect(client.ip, client.port)
    @info "Connected to AST server in $dt seconds." client.conn
end

"""
Connects client with server at new location.
"""
function connect!(client::ASTClient, ip::IPAddr, port::Int64)
    client.ip = ip
    client.port = port
    connect!(client)
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
    return response[:z]
end

function CommonRLInterface.reset!(client::ASTClient)
    client.verbose && @info "Sending request to server: `reset!`"
    call(client, CommonRLInterface.reset!)
end

function CommonRLInterface.actions(client::ASTClient)
    client.verbose && @info "Sending request to server: `actions`"
    Dirac(call(client, CommonRLInterface.actions)) # returns pseudo-distribution
end

function CommonRLInterface.observe(client::ASTClient)
    client.verbose && @info "Sending request to server: `observe`"
    call(client, CommonRLInterface.observe)
end

function CommonRLInterface.act!(client::ASTClient, action::Vector{<:Real})
    client.verbose && @info "Sending request to server: `act!`"
    call(client, CommonRLInterface.act!, action)
end

function CommonRLInterface.terminated(client::ASTClient)
    client.verbose && @info "Sending request to server: `terminated`"
    call(client, CommonRLInterface.terminated)
end

"""
Simulation-side server. Interacts with client via TCP.
"""
Base.@kwdef mutable struct ASTServer <: CommonRLInterface.AbstractEnv
    ip::IPAddr                              = getipaddr()  # address(es) to listen on
    port::Int64                             = 2000         # port to listen on
    serv::Union{Sockets.TCPServer, Nothing} = nothing
    mdp::ASTMDP
    verbose::Bool                           = false
end

"""
Constructor for server.
"""
function ASTServer(mdp::ASTMDP, ip::IPAddr, port::Int64; kwargs...)
    ASTServer(; mdp=mdp, ip=ip, port=port, kwargs...)
end

"""
Disconnects server.
"""
function disconnect!(server::ASTServer)
    if server.serv !== nothing
        close(server.serv)
        server.serv = nothing
    end
end

"""
Listens for incoming requests and executes function calls on MDP.
Can handle multiple client connections asynchronously.
"""
function run(server::ASTServer)
    @async while true
        conn = accept(server.serv)
        @info "Connected to AST client." conn
        @async while true
            request = BSON.load(conn)
            sym = FMAP[request[:f]]
            server.verbose && @info "Received request from client: `$sym`"
            f = getproperty(CommonRLInterface, sym)
            args = request[:a]

            z = f(server.mdp, args...)
            response = Dict(:z => sym == :actions ? rand(z) : z) # performs rand server-side
            server.verbose && @info "Sending response to client."
            bson(conn, response)
        end
    end
end

"""
Connects server.
"""
function connect!(server::ASTServer)
    disconnect!(server)
    server.serv = listen(server.ip, server.port)
    run(server)
end

"""
Connects server at new location.
"""
function connect!(server::ASTServer, ip::IPAddr, port::Int64)
    server.ip = ip
    server.port = port
    connect!(server)
end
