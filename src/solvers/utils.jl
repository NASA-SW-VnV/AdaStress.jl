"""
Top-k priority queueing. Deletes least significant keys if length exceeds `k`.
"""
function DataStructures.enqueue!(pq::PriorityQueue, key, value, k::Int64)
    pq[key] = value
    while length(pq) > k
        delete!(pq, first(keys(pq)))
    end
end
