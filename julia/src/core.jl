# The TTP-D model from the paper. Node 0 is the depot and node n + 1 is the depot again on return;
# a node v lives at position v + 1 in `dist`, `p`, `w` and `z`.

const EPS = 1e-9
const Sortie = NTuple{3,Int}          # (launch, customer, landing)

struct Problem
    n::Int
    dist::Matrix{Float64}
    p::Vector{Float64}
    w::Vector{Float64}
    cap::Float64
    vmax::Float64
    vmin::Float64
    vdrone::Float64
    rent::Float64
    endurance::Float64                # Inf is unlimited
    slope::Float64
end

Problem(n, dist, p, w, cap, vmax, vmin, vdrone, rent, endurance) =
    Problem(n, dist, p, w, cap, vmax, vmin, vdrone, rent, endurance, (vmax - vmin) / cap)

d(P::Problem, a::Int, b::Int) = @inbounds P.dist[a+1, b+1]

"""The objective of a plan, or `nothing` if it breaks a rule."""
function evaluate(P::Problem, truck::Vector{Int}, sorties::Vector{Sortie}, z::Vector{Bool})
    timing = mission(P, truck, sorties, z)
    timing === nothing ? nothing : timing[2] - P.rent * timing[1]
end

"""The finishing time and profit of a plan, or `nothing` if it breaks a rule."""
function mission(P::Problem, truck::Vector{Int}, sorties::Vector{Sortie}, z::Vector{Bool})
    n = P.n
    m = length(truck) + 2
    pos = zeros(Int, n + 2)           # position on the route, 0 when the node is not on it
    pos[1] = 1
    for (i, v) in enumerate(truck)
        pos[v+1] = i + 1
    end
    pos[n+2] = m

    order = sort(sorties; by=s -> pos[s[1]+1])
    launch = zeros(Int, m)
    landing = zeros(Int, m)
    last_end = 1
    for (i, (L, D, J)) in enumerate(order)
        a, b = pos[L+1], pos[J+1]
        (a == 0 || b == 0 || a >= b || a < last_end) && return nothing
        d(P, L, D) + d(P, D, J) > P.endurance + EPS && return nothing
        last_end = b
        launch[a] = i
        landing[b] = i
    end

    cap = P.cap + EPS
    clock = 0.0
    load = 0.0
    profit = 0.0
    left = zeros(length(order))
    for i in 1:m
        node = i == 1 ? 0 : i == m ? n + 1 : truck[i-1]
        s = landing[i]
        if s != 0
            L, D, _ = order[s]
            arrival = left[s] + (d(P, L, D) + d(P, D, node)) / P.vdrone
            arrival > clock && (clock = arrival)
            if z[D+1]
                load += P.w[D+1]
                load > cap && return nothing
                profit += P.p[D+1]
            end
        end
        if 0 < node <= n && z[node+1]
            load += P.w[node+1]
            load > cap && return nothing
            profit += P.p[node+1]
        end
        s = launch[i]
        s != 0 && (left[s] = clock)
        if i < m
            next = i + 1 == m ? n + 1 : truck[i]
            clock += d(P, node, next) / (P.vmax - load * P.slope)
        end
    end
    return clock, profit
end

"""Why a plan is infeasible, or `nothing` if it is sound."""
function problem_with(P::Problem, truck, sorties, z)
    targets = [s[2] for s in sorties]
    allunique(truck) || return "duplicate truck customers"
    allunique(targets) || return "duplicate flight customers"
    sort!(vcat(truck, targets)) == collect(1:P.n) || return "truck and drone do not cover the customers"
    (z[1] || z[P.n+2]) && return "an item taken at the depot"
    evaluate(P, truck, sorties, z) === nothing && return "the plan breaks a timing, endurance or capacity rule"
    return nothing
end

const SCHEDULE_CAP = 12

"""The best flights for these drone customers along a fixed truck route, by dynamic programming."""
function schedule(P::Problem, truck::Vector{Int}, targets::Vector{Int}, z::Vector{Bool})
    k = length(targets)
    if k == 0
        v = evaluate(P, truck, Sortie[], z)
        return v === nothing ? nothing : Sortie[]
    end
    k > SCHEDULE_CAP && return nothing
    n = P.n
    route = vcat(0, truck, n + 1)
    m = length(route)
    sum(P.w[i+1] for i in 1:n if z[i+1]; init=0.0) > P.cap + EPS && return nothing

    prefix = zeros(m)
    acc = 0.0
    for (i, v) in enumerate(route)
        0 < v <= n && z[v+1] && (acc += P.w[v+1])
        prefix[i] = acc
    end
    wd = [z[t+1] ? P.w[t+1] : 0.0 for t in targets]
    full = (1 << k) - 1
    carried = zeros(full + 1)
    for mask in 1:full
        low = mask & -mask
        carried[mask+1] = carried[(mask⊻low)+1] + wd[trailing_zeros(low)+1]
    end

    best = [Dict{Int,Tuple{Float64,Int,Int,Sortie}}() for _ in 1:m]    # time, previous position, previous mask, flight
    best[1][0] = (0.0, 0, 0, (-1, -1, -1))
    arrive = zeros(m)
    for i in 1:m-1
        isempty(best[i]) && continue
        here = route[i]
        for (mask, entry) in collect(best[i])
            clock = entry[1]
            load = carried[mask+1]
            t = clock
            for j in i+1:m
                t += d(P, route[j-1], route[j]) / (P.vmax - (prefix[j-1] + load) * P.slope)
                arrive[j] = t
            end
            old = get(best[i+1], mask, nothing)
            if old === nothing || arrive[i+1] < old[1] - 1e-12
                best[i+1][mask] = (arrive[i+1], i, mask, (-1, -1, -1))
            end
            for b in 0:k-1
                bit = 1 << b
                mask & bit != 0 && continue
                D = targets[b+1]
                out = d(P, here, D)
                next = mask | bit
                for j in i+1:m
                    round_trip = out + d(P, D, route[j])
                    round_trip > P.endurance + EPS && continue
                    t = max(arrive[j], clock + round_trip / P.vdrone)
                    old = get(best[j], next, nothing)
                    if old === nothing || t < old[1] - 1e-12
                        best[j][next] = (t, i, mask, (here, D, route[j]))
                    end
                end
            end
        end
    end

    haskey(best[m], full) || return nothing
    flights = Sortie[]
    i, mask = m, full
    while i != 1 || mask != 0
        _, pi, pmask, flight = best[i][mask]
        flight[1] >= 0 && push!(flights, flight)
        i, mask = pi, pmask
    end
    return reverse!(flights)
end
