# cube: one call for four problems. What you leave out decides which.

const SPEED, SLOWEST, DRONE_FACTOR = 1.0, 0.1, 2.0
const NO_FLIGHT = 1e-9        # an endurance no flight can meet: this is what switches the drone off

"""A finished plan. Node numbers count from 0 at the depot, the same as Python, so customer `i` is `nodes[i + 1]`."""
struct Plan
    problem::String           # "TSP", "TSP-D", "TTP" or "TTP-D"
    route::Vector{Int}        # the truck's path, from the depot back to it
    flights::Vector{Sortie}   # drone flights as (launch, customer, landing)
    items::Vector{Int}        # customers whose item is taken
    time::Float64             # when everyone is back
    profit::Float64           # value of the items taken
    objective::Float64        # profit minus rent times time
    seconds::Float64          # how long the solver ran
end

function Base.show(io::IO, plan::Plan)
    money = round(plan.objective; digits=2)
    parts = ["time $(round(plan.time; digits=2))"]
    plan.problem in ("TTP", "TTP-D") && (parts = ["objective $money", "profit $(round(plan.profit; digits=2))", parts..., "$(length(plan.items)) items"])
    plan.problem in ("TSP-D", "TTP-D") && push!(parts, "$(length(plan.flights)) flights")
    print(io, plan.problem, " plan: ", join(parts, ", "))
end

function column(values, n, name)
    length(values) == n || throw(ArgumentError("$name has $(length(values)) values for $n customers; the depot takes none"))
    out = Float64.(collect(values))
    all(v -> isfinite(v) && v >= 0, out) || throw(ArgumentError("$name must be finite and not negative"))
    return out
end

"""
    cube(nodes, profits=nothing, weights=nothing; drone=false, endurance=nothing, capacity=nothing,
         rent=nothing, drone_speed=nothing, seconds=nothing, seed=0, dist=nothing)

Plan with the metaheuristic. Leave out the items for a TSP or TSP-D, and leave out the drone for a TSP or TTP.

- `nodes`: (x, y) pairs, depot first.
- `profits`, `weights`: one value per customer, depot left out. Give both or neither.
- `drone`: fly a drone from the truck.
- `endurance`: longest flight, launch to customer to landing. Unlimited by default; setting it turns the drone on.
- `capacity`: what the truck can carry. Half the total weight by default.
- `rent`: cost per unit of time. By default, the rate at which a plain tour costs half of all the profit.
- `drone_speed`: twice the empty truck's speed by default.
- `seconds`: search time. Half a second per customer by default; the paper gives it 15.
- `seed`: random seed for the search.
- `dist`: your own distance matrix, depot first, if straight lines are wrong.
"""
function cube(nodes, profits=nothing, weights=nothing; drone::Bool=false, endurance=nothing, capacity=nothing,
                  rent=nothing, drone_speed=nothing, seconds=nothing, seed::Integer=0, dist=nothing)
    started = time()
    (profits === nothing) == (weights === nothing) || throw(ArgumentError("give profits and weights together, or neither"))
    length(nodes) >= 2 || throw(ArgumentError("nodes needs the depot and at least one customer"))
    points = [(Float64(x), Float64(y)) for (x, y) in nodes]
    all(p -> all(isfinite, p), points) || throw(ArgumentError("each node must be a finite (x, y) pair"))
    n = length(points) - 1

    order = vcat(points, points[1:1])
    table = if dist === nothing
        [hypot(a[1] - b[1], a[2] - b[2]) for a in order, b in order]
    else
        size(dist) == (n + 1, n + 1) || throw(ArgumentError("dist must be $(n + 1) by $(n + 1), depot first"))
        keep = vcat(1:n+1, 1)
        Float64.(dist[keep, keep])
    end

    items = profits !== nothing
    p = items ? vcat(0.0, column(profits, n, "profits"), 0.0) : zeros(n + 2)
    w = items ? vcat(0.0, column(weights, n, "weights"), 0.0) : zeros(n + 2)
    flying = drone || endurance !== nothing || drone_speed !== nothing
    endurance !== nothing && endurance <= 0 && throw(ArgumentError("endurance must be positive; leave the drone out for no drone"))
    cap = capacity === nothing ? (sum(w) > 0 ? sum(w) / 2 : 1.0) : Float64(capacity)
    cap > 0 || throw(ArgumentError("capacity must be positive"))
    fly = flying ? (endurance === nothing ? Inf : Float64(endurance)) : NO_FLIGHT
    vdrone = drone_speed === nothing ? SPEED * DRONE_FACTOR : Float64(drone_speed)

    charge = if rent !== nothing
        Float64(rent)
    elseif sum(p) > 0        # the rate at which driving a plain tour costs half of all the profit
        tour = vcat(0, nearest_route(Problem(n, table, p, w, cap, SPEED, SLOWEST, vdrone, 1.0, fly), 1:n), n + 1)
        sum(p) / 2 / max(sum(table[a+1, b+1] for (a, b) in zip(tour, tour[2:end])) / SPEED, 1e-9)
    else
        1.0
    end
    charge >= 0 || throw(ArgumentError("rent must not be negative"))

    P = Problem(n, table, p, w, cap, SPEED, SLOWEST, vdrone, charge, fly)
    budget = seconds === nothing ? max(1.0, 0.5 * n) : Float64(seconds)
    budget > 0 || throw(ArgumentError("seconds must be positive"))
    truck, sorties, z, _ = metaheuristic(P, budget; seed=seed)

    home(v) = v == n + 1 ? 0 : v
    finish, profit = mission(P, truck, sorties, z)
    name = flying ? (items ? "TTP-D" : "TSP-D") : (items ? "TTP" : "TSP")
    return Plan(name, vcat(0, truck, 0), [(home(a), b, home(c)) for (a, b, c) in sorties],
                [i for i in 1:n if z[i+1]], finish, profit, profit - charge * finish, time() - started)
end
