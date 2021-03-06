# ******************************************************************************************
# Notices:
#
# Copyright © 2022 United States Government as represented by the Administrator of the
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

#TODO: robustness, isopen checks, reconnection

FLAGS = [:_echo, :_info, :reset!, :actions, :observe, :act!, :terminated]
FMAP = Bijection(Dict(UInt8(i) => f for (i, f) in enumerate(FLAGS))) # maps flags <-> bytes

"""
    ASTClient

Solver-side client. Interacts with server via TCP.
"""
Base.@kwdef mutable struct ASTClient
    ip::IPAddr                      = ip"::1" # address(es) to connect to
    port::Int64                     = 1812    # port to connect to
    conn::Union{TCPSocket, Nothing} = nothing
    tunnel::Bool                    = false
    verbose::Bool                   = false
    caching::Bool                   = false   # accelerates episodic problems
    cache::Vector                   = []
end

"""
    RemoteASTMDP{S<:State, A<:Action} <: AbstractASTMDP{S, A}

MDP object that virtually references remote MDP. Autogenerated by `ASTClient`.
"""
Base.@kwdef mutable struct RemoteASTMDP{S<:State, A<:Action} <: AbstractASTMDP{S, A}
    client::ASTClient
    episodic::Bool   = false
    num_steps::Int64 = 0
end

"""
Exception type for server-side error. Allows exceptions to propogate to client to avoid
hanging connection.
"""
Base.@kwdef struct ASTServerError <: Exception
    msg::String = "ASTServer encountered an error."
end
Base.showerror(io::IO, e::ASTServerError) = print(io, e.msg)

"""
Perform function call on server MDP. Call sends request to server with provided function
and arguments and receives return value. Blocks until complete.
"""
function call(client::ASTClient, f::Function, args...; kwargs...)
    flag = Symbol(f)
    request = Dict(:f => FMAP(flag), :a => args, :k => NamedTuple(kwargs))
    bson(client.conn, request)
    client.verbose && @info "Sending request to server:" request
    response = BSON.load(client.conn)
    client.verbose && @info "Received response from server:" response
    haskey(response, :e) && response[:e] && throw(ASTServerError())
    return response[:r]
end

function reset!(mdp::RemoteASTMDP)
    mdp.client.caching && empty!(mdp.client.cache)
    sync(mdp.client)
    call(mdp.client, reset!)
end

function actions(mdp::RemoteASTMDP)
    action_type(mdp) == SeedAction ? UInt32 : Dirac(call(mdp.client, actions))
end

function observe(mdp::RemoteASTMDP)
    call(mdp.client, observe)
end

#TODO: check for off-by-one errors
function act!(mdp::RemoteASTMDP, action)
    if mdp.client.caching
        cache = mdp.client.cache
        done = length(cache) == mdp.num_steps
        push!(cache, action)
        return done ? call(mdp.client, act!, cache; batch=true) : 0.0
    else
        return call(mdp.client, act!, action)
    end
end

function terminated(mdp::RemoteASTMDP)
    mdp.client.caching ? length(mdp.client.cache) >= mdp.num_steps : call(mdp.client, terminated)
end

"""
    ping(client::ASTClient)

Request empty echo from ASTServer, returning approximate round-trip time in seconds.
"""
ping(client::ASTClient) = @elapsed call(client, _echo, nothing)

"""
    generate_mdp(client::ASTClient)

Query remote MDP and generate `RemoteASTMDP` with matching attributes to interface with
appropriate solvers.
"""
function generate_mdp(client::ASTClient)
    attr = call(client, _info)
    S = getproperty(Interface, attr[:S])
    A = getproperty(Interface, attr[:A])
    return RemoteASTMDP{S, A}(client, attr[:episodic], attr[:num_steps])
end

"""
Read from socket until communication is synchronized. Typically unnecessary, but mitigates
issue in which IO between client and server can become mismatched.
"""
function sync(client::ASTClient)
    seed = rand(RandomDevice(), UInt32)
    r = call(client, _echo, seed)
    while r != seed
        r = BSON.load(client.conn)[:r]
    end
    return
end

"""
    connect!(client::ASTClient; remote::String="", remote_port::Int64=1812)

Connect client to server, optionally through SSH tunnel. Optional keyword argument `remote`
should be of the form `user@machine`.
"""
function connect!(client::ASTClient; remote::String="", remote_port::Int64=1812)
    disconnect!(client)
    !isempty(remote) && open_tunnel(client, remote, remote_port)
    client.conn = connect(client.ip, client.port)
    Sockets.nagle(client.conn, false)
    t_ms = Int64(1000 * round(ping(client); digits=3))
    @info "ASTServer responded in $t_ms milliseconds."
    return
end

"""
    disconnect!(client::ASTClient)

Disconnect client from server.
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
