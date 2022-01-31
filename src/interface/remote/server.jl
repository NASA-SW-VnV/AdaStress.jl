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
