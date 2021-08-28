
"""
Anonymous state node defined by unique chain of preceding actions.
Search tree is composed of singly- and doubly-linked nodes.
"""
Base.@kwdef mutable struct Node{A}
    n::Int64                                 = 0
    actions::Dict{A, Node{A}}                = Dict{A, Node{A}}()
    q::Float64                               = 0.0
    parent::Union{Pair{A, Node{A}}, Nothing} = nothing
    new::Bool                                = true
end

"""
Returns whether node is a head (no parent).
"""
ishead(node::Node) = node.parent === nothing

"""
Returns whether node is a leaf (no children).
"""
isleaf(node::Node) = isempty(node.actions)

"""
Returns iterable collection of children.
"""
children(node::Node) = values(node.actions)

"""
Takes an action from a given state, adding and returning new child.
For transient action chains, such as rollouts, set forward link to `false`;
nodes will be deleted automatically by garbage collector when leaf goes out of scope.
"""
function add(node::Node{A}, a::A; forward::Bool=true, backward::Bool=true) where A
    child = Node{A}()
    if forward
        node.actions[a] = child
    end
    if backward
        child.parent = (a => node)
    end
    return child
end

"""
Returns action chain from head to current node.
"""
function trace_back(node::Node{A}) where A
    ishead(node) ? A[] : push!(trace_back(node.parent[2]), node.parent[1])
end

"""
Maximum depth of tree.
"""
max_depth(tree::Node) = isleaf(tree) ? 0 : 1 + maximum(max_depth.(children(tree)))

"""
Total number of nodes in tree.
"""
total_size(tree::Node) = isleaf(tree) ? 1 : 1 + sum(total_size.(children(tree)))
