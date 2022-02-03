"""
Delete least significant keys if length exceeds `k`. Facilitates top-k priority queueing.
"""
function DataStructures.enqueue!(pq::PriorityQueue, key, value, k::Int64)
    pq[key] = value
    while length(pq) > k
        delete!(pq, first(keys(pq)))
    end
end

function replay!(mdp::Interface.AbstractASTMDP, result::LocalResult)
    as = getproperty(result, fieldnames(typeof(result))[1])
    Interface.reset!(mdp)
    for a in as
        Interface.act!(mdp, a)
    end
    return Interface.isevent(mdp.sim)
end

function replay!(mdp::Interface.AbstractASTMDP, result::GlobalResult)
    Interface.reset!(mdp)
    while !Interface.terminated(mdp)
        a = result(Interface.observe(mdp))
        Interface.act!(mdp, a)
    end
    return Interface.isevent(mdp.sim)
end

struct RandomPolicy end

function replay!(mdp::Interface.AbstractASTMDP, ::RandomPolicy)
    Interface.reset!(mdp)
    while !Interface.terminated(mdp)
        a = rand(Interface.actions(mdp))
        Interface.act!(mdp, a)
    end
    return Interface.isevent(mdp.sim)
end

struct NullPolicy end

function replay!(mdp::Interface.AbstractASTMDP, ::NullPolicy)
    Interface.reset!(mdp)
    while !Interface.terminated(mdp)
        a = zero(rand(Interface.actions(mdp)))
        Interface.act!(mdp, a)
    end
    return Interface.isevent(mdp.sim)
end
