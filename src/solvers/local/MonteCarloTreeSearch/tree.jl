"""
Anonymous state node defined by unique chain of preceding actions. Search tree is composed
of singly- and doubly-linked nodes.
"""
Base.@kwdef mutable struct Node{A}
    n::Int64                                              = 0
    q::Float64                                            = 0.0
    transitions::PriorityQueue{Pair{A, Node{A}}, Float64} = PriorityQueue{Pair{A, Node{A}}, Float64}(Base.Order.Reverse)
    parent::Union{Pair{A, Node{A}}, Nothing}              = nothing
end

"""
Return whether node is a root (no parent).
"""
isroot(node::Node) = node.parent === nothing

"""
Return whether node is a leaf (no children).
"""
isleaf(node::Node) = isempty(node.transitions)

"""
Return unordered collection of children. Uses PQ backend to avoid unnecessary sort.
"""
children(node::Node) = last.(first.(node.transitions.xs))

"""
Return top-scoring action and next state.
"""
top_transition(node::Node) = first(peek(node.transitions))

"""
Take an action from a given state, adding and returning new child. For transient action
chains, such as rollouts, set forward link to `false`; nodes are deleted automatically by
garbage collector when leaf goes out of scope.
"""
function add(node::Node{A}, a::A; forward::Bool=true, backward::Bool=true) where A
    child = Node{A}()
    if forward
        node.transitions[a => child] = 0.0
    end
    if backward
        child.parent = a => node
    end
    return child
end

"""
Return action chain from root to current node.
"""
function trace(node::Node{A}) where A
    isroot(node) ? A[] : push!(trace(node.parent[2]), node.parent[1])
end

"""
Return maximum depth of tree.
"""
max_depth(tree::Node) = isleaf(tree) ? 0 : 1 + maximum(max_depth.(children(tree)))

"""
Return total number of nodes in tree.
"""
total_size(tree::Node) = isleaf(tree) ? 1 : 1 + sum(total_size.(children(tree)))
