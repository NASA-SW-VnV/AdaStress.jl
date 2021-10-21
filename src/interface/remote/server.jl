
"""
Optional private token for additional security. Use of token prompts seed rehashing to
obstruct client-side interpretability.
"""
struct Token
    data::Any
end
Base.show(io::IO, ::Token) = print(io, "Token()")

"""
Salt random seed with private token for additional security.
"""
reseed(seed::UInt32, token::Token) = UInt32(hash((seed, token.data)) % typemax(UInt32))

"""
    ASTServer

Simulation-side server. Interacts with client via TCP.
"""
Base.@kwdef mutable struct ASTServer
    ip::IPAddr                              = ip"::1"  # address(es) to listen on
    port::Int64                             = 1812     # port to listen on
    serv::Union{Sockets.TCPServer, Nothing} = nothing
    mdp::ASTMDP
    token::Union{Token, Nothing}            = nothing
    tunnel::Bool                            = false
    verbose::Bool                           = false
    presample::Bool                         = true
end

"""
    ASTServer(mdp::ASTMDP; kwargs...)

Constructor for server.
"""
function ASTServer(mdp::ASTMDP; kwargs...)
    ASTServer(; mdp=mdp, kwargs...)
end

"""
    set_token(server::ASTServer, token)

Set private token to encrypt seeds. Recommended to use `set_password` instead, to avoid
leaking token through side effects.
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
    set_password(server::ASTServer)

Set private token to string input (password). Avoids explicit representation of password in
storage or side effects.
"""
function set_password(server::ASTServer)
    io = Base.getpass("Enter password")
    Base.shred!(io) do io
        set_token(server, read(io, String))
    end
    return
end

"""
Respond to echo request from ASTClient.
"""
_echo(::ASTMDP, x) = x

"""
Respond to information request from ASTClient.
"""
function _info(mdp::ASTMDP{S, A}) where {S, A}
    Dict(
        :S => Symbol("$S"),
        :A => Symbol("$A"),
        :episodic => mdp.episodic,
        :num_steps => mdp.num_steps
    )
end

"""
Process simulation request from client and construct response.
"""
function respond(server::ASTServer, request::Dict)
    # interpret request
    sym = FMAP[request[:f]]
    f = getproperty(Interface, sym)
    args = request[:a]
    kwargs = request[:k]

    # reseeding
    if server.token !== nothing && sym == :act! && action_type(server.mdp) == SeedAction
        seed = reseed(args[1], server.token)
        args = (seed,)
    end

    if haskey(kwargs, :batch) && kwargs.batch
        r = sum(f.(Ref(server.mdp), args...)) # batch processing
    else
        r = f(server.mdp, args...)
    end

    # server-side sampling
    if server.presample && sym == :actions && action_type(server.mdp) == SampleAction
        r = rand(r; flat=true)
    end

    return Dict(:r => r)
end

"""
Listen for incoming requests and execute function calls on MDP. Can handle multiple client
connections asynchronously, but server currently holds a single MDP instance.
"""
function run(server::ASTServer)
    @async while true
        # listen for incoming connections
        conn = accept(server.serv)
        Sockets.nagle(conn, false)
        @info "Connected to AST client." conn
        @async while true
            request = BSON.load(conn)
            server.verbose && @info "Received request from client:" request
            response = try
                respond(server, request)
            catch e
                bson(conn, Dict(:e => true))
                throw(e)
            end
            bson(conn, response)
            server.verbose && @info "Sent response to client." response
        end
    end
end

"""
    connect!(server::ASTServer; remote::String="", remote_port::Int64=1812)

Connect server, optionally through SSH tunnel. Optional keyword argument `remote` should be
of the form `user@machine`.
"""
function connect!(server::ASTServer; remote::String="", remote_port::Int64=1812)
    disconnect!(server)
    !isempty(remote) && open_tunnel(server, remote, remote_port)
    server.serv = listen(server.ip, server.port)
    run(server)
    return
end

"""
    disconnect!(server::ASTServer)

Disconnect server.
"""
function disconnect!(server::ASTServer)
    if server.serv !== nothing
        close(server.serv)
        server.serv = nothing
    end
    server.tunnel && close_tunnel()
    server.tunnel = false
    return
end
