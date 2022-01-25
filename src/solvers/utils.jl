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
end

function replay!(mdp::Interface.AbstractASTMDP, result::GlobalResult)
    Interface.reset!(mdp)
    while Interface.terminated(mdp)
        a = result(AdaStress.observe(mdp))
        Interface.act!(mdp, a)
    end
end
