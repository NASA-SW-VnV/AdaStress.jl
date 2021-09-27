
"""
Optional private token for extra security.
Forces seed rehashing to remove client-side interpretability.
"""
struct Token
    data::Any
end
Base.show(io::IO, ::Token) = print(io, "Token()")

"""
Salts random seed with private token for extra security.
"""
reseed(seed::UInt32, token::Token) = UInt32(hash((seed, token.data)) % typemax(UInt32))

"""
Simulation-side server. Interacts with client via TCP.
"""
Base.@kwdef mutable struct ASTServer
    ip::IPAddr                              = getipaddr()  # address(es) to listen on
    port::Int64                             = 2000         # port to listen on
    serv::Union{Sockets.TCPServer, Nothing} = nothing
    mdp::ASTMDP
    token::Union{Token, Nothing}            = nothing
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
Sets private token.
"""
function set_token(server::ASTServer, token)
    if action_type(server.mdp) == SeedAction
        server.token = Token(token)
        @info "Private token set."
    else
        @info "Private token can only be set for seed-action (blackbox) simulators."
    end
    return
end

"""
Sets private token to string input (password).
Avoids explicit representation of password in storage or side effects.
"""
function set_password(server::ASTServer)
    io = Base.getpass("Enter password")
    Base.shred!(io) do io
        set_token(server, read(io, String))
    end
    return
end

"""
Listens for incoming requests and executes function calls on MDP.
Can handle multiple client connections asynchronously.
"""
function run(server::ASTServer)
    @async while true
        # Listen for incoming connections.
        conn = accept(server.serv)
        @info "Connected to AST client." conn
        @async while true
            # Interpret request.
            request = BSON.load(conn)
            sym = FMAP[request[:f]]
            server.verbose && @info "Received request from client: `$sym`"
            f = getproperty(Interface, sym)
            args = request[:a]

            # Process response
            if server.token !== nothing && sym == :act! && action_type(server.mdp) == SeedAction
                seed = reseed(args[1], server.token)
                args = (seed,)
            end
            r = f(server.mdp, args...)
            if sym == :actions && action_type(server.mdp) == SampleAction
                r = Dirac(rand(r)) # perform rand server-side
            end

            # Send response.
            response = Dict(:r => r)
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