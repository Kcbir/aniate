"""
    Aniate

Open source combinatorics solvers.

    using Aniate

    nodes = [(0, 0), (4, 1), (5, 4), (1, 5), (2, 2), (3, 3)]   # depot first
    profits, weights = [50, 60, 70, 80, 40], [2, 3, 4, 5, 2]

    cube(nodes)                                 # TSP: a truck
    cube(nodes; drone=true)                     # TSP-D: a truck and a drone
    cube(nodes, profits, weights)               # TTP: a truck and a knapsack
    cube(nodes, profits, weights; drone=true)   # TTP-D: all three
"""
module Aniate

using Random

export cube

include("core.jl")
include("moves.jl")
include("search.jl")
include("route.jl")

end
