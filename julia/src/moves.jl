# The ten neighbourhoods of the paper's descent, and the single random move of its search.
# Each neighbourhood calls `visit(truck, sorties, z)` per candidate, and stops when that returns true.

anchors(sorties) = Set(x for s in sorties for x in (s[1], s[3]))

"""Flights whose ends swapped order on a new route, turned the right way round, or `nothing`."""
function repaired(P::Problem, truck::Vector{Int}, sorties::Vector{Sortie})
    pos = zeros(Int, P.n + 2)
    pos[1] = 1
    for (i, v) in enumerate(truck)
        pos[v+1] = i + 1
    end
    pos[P.n+2] = length(truck) + 2
    changed = false
    out = copy(sorties)
    for (i, (L, D, J)) in enumerate(sorties)
        (pos[L+1] == 0 || pos[J+1] == 0) && return nothing
        if pos[L+1] > pos[J+1]
            out[i] = (J, D, L)
            changed = true
        end
    end
    return changed ? out : nothing
end

ordered(items, rng) = rng === nothing ? items : shuffle(rng, collect(items))
flipped(z, i) = (c = copy(z); c[i+1] = !c[i+1]; c)
without(v, i) = deleteat!(copy(v), i)
replaced(v, i, x) = (c = copy(v); c[i] = x; c)

function with_repair(visit, P, truck, sorties, z)
    visit(truck, sorties, z) && return true
    fixed = repaired(P, truck, sorties)
    return fixed !== nothing && visit(truck, fixed, z)
end

function flip_items(visit, P, truck, sorties, z, rng)
    for i in ordered(1:P.n, rng)
        visit(truck, sorties, flipped(z, i)) && return true
    end
    false
end

function swap_truck(visit, P, truck, sorties, z, rng)
    m = length(truck)
    for a in ordered(1:m-1, rng), b in a+1:m
        t = copy(truck)
        t[a], t[b] = t[b], t[a]
        with_repair(visit, P, t, sorties, z) && return true
    end
    false
end

function two_opt(visit, P, truck, sorties, z, rng)
    m = length(truck)
    for a in ordered(1:m-1, rng), b in a+1:m
        t = copy(truck)
        reverse!(t, a, b)
        with_repair(visit, P, t, sorties, z) && return true
    end
    false
end

function or_opt(visit, P, truck, sorties, z, rng)
    m = length(truck)
    for len in 1:3
        len > m && break
        for a in ordered(1:m-len+1, rng)
            segment = truck[a:a+len-1]
            rest = vcat(truck[1:a-1], truck[a+len:end])
            for b in 0:length(rest)
                t = vcat(rest[1:b], segment, rest[b+1:end])
                t == truck && continue
                with_repair(visit, P, t, sorties, z) && return true
            end
        end
    end
    false
end

function reschedule(visit, P, truck, sorties, z, rng)
    flights = schedule(P, truck, [s[2] for s in sorties], z)
    flights !== nothing && flights != sorties && visit(truck, flights, z) && return true
    false
end

function reanchor(visit, P, truck, sorties, z, rng)
    route = vcat(0, truck, P.n + 1)
    m = length(route)
    for (i, (L, D, J)) in enumerate(sorties)
        other = flipped(z, D)
        for a in ordered(1:m-1, rng), b in a+1:m
            s = replaced(sorties, i, (route[a], D, route[b]))
            (route[a] != L || route[b] != J) && visit(truck, s, z) && return true
            visit(truck, s, other) && return true
        end
    end
    false
end

function swap_truck_drone(visit, P, truck, sorties, z, rng)
    fixed = anchors(sorties)
    for (ti, t) in enumerate(truck)
        t in fixed && continue
        for (si, (L, D, J)) in enumerate(sorties)
            nt = replaced(truck, ti, D)
            ns = replaced(sorties, si, (L, t, J))
            for zt in (false, true), zd in (false, true)
                nz = copy(z)
                nz[t+1], nz[D+1] = zt, zd
                visit(nt, ns, nz) && return true
            end
        end
    end
    false
end

function drone_to_truck(visit, P, truck, sorties, z, rng)
    for (i, (L, D, J)) in enumerate(sorties)
        s = without(sorties, i)
        other = flipped(z, D)
        for b in 0:length(truck)
            t = vcat(truck[1:b], D, truck[b+1:end])
            visit(t, s, z) && return true
            visit(t, s, other) && return true
        end
    end
    false
end

function truck_to_drone(visit, P, truck, sorties, z, rng)
    fixed = anchors(sorties)
    for t in ordered([x for x in truck if !(x in fixed)], rng)
        rest = filter(!=(t), truck)
        route = vcat(0, rest, P.n + 1)
        m = length(route)
        other = flipped(z, t)
        for a in 1:m-1, b in a+1:m
            s = vcat(sorties, (route[a], t, route[b]))
            visit(rest, s, z) && return true
            visit(rest, s, other) && return true
        end
    end
    false
end

const FLIP_SCHEDULE_CAP = 10

function flip_and_schedule(visit, P, truck, sorties, z, rng)
    targets = [s[2] for s in sorties]
    length(targets) + 1 > FLIP_SCHEDULE_CAP && return false
    for (i, t) in enumerate(truck)
        rest = without(truck, i)
        more = vcat(targets, t)
        for nz in (z, flipped(z, t))
            flights = schedule(P, rest, more, nz)
            flights !== nothing && visit(rest, flights, nz) && return true
        end
    end
    false
end

const NEIGHBOURHOODS = (flip_items, swap_truck, two_opt, or_opt, reschedule, reanchor, swap_truck_drone,
                        drone_to_truck, truck_to_drone, flip_and_schedule)

"""First-improvement descent through the neighbourhoods. Stops at `deadline`, in `time()` seconds."""
function descend(P, truck, sorties, z, value, rng, deadline, cap)
    k = 1
    shuffled = cap === nothing ? nothing : rng
    while k <= length(NEIGHBOURHOODS)
        time() >= deadline && break
        seen = 0
        improved = false
        expired = false
        NEIGHBOURHOODS[k](P, truck, sorties, z, shuffled) do t, s, nz
            seen += 1
            v = evaluate(P, t, s, nz)
            if v !== nothing && v > value + EPS
                truck, sorties, z, value = t, s, nz, v
                improved = true
                return true
            end
            if time() >= deadline
                expired = true
                return true
            end
            return cap !== nothing && seen >= cap
        end
        expired && break
        k = improved ? 1 : k + 1
    end
    return truck, sorties, z, value
end

"""One random feasible move, or `nothing` after thirty tries."""
function random_move(P, truck, sorties, z, rng)
    for _ in 1:30
        op = rand(rng, 0:5)
        t, s, nz = truck, sorties, z
        route_of(r) = vcat(0, r, P.n + 1)
        ends(m) = (a = rand(rng, 1:m-1); (a, rand(rng, a+1:m)))
        if op == 0 && P.n > 0
            nz = flipped(z, rand(rng, 1:P.n))
        elseif op == 1 && !isempty(sorties)
            i = rand(rng, eachindex(sorties))
            route = route_of(truck)
            a, b = ends(length(route))
            s = replaced(sorties, i, (route[a], sorties[i][2], route[b]))
        elseif op == 2 && !isempty(truck)
            c = rand(rng, truck)
            c in anchors(sorties) && continue
            t = filter(!=(c), truck)
            route = route_of(t)
            a, b = ends(length(route))
            s = vcat(sorties, (route[a], c, route[b]))
        elseif op == 3 && !isempty(sorties)
            i = rand(rng, eachindex(sorties))
            D = sorties[i][2]
            s = without(sorties, i)
            b = rand(rng, 0:length(truck))
            t = vcat(truck[1:b], D, truck[b+1:end])
        elseif op == 4 && length(truck) >= 2
            a = rand(rng, eachindex(truck))
            c = truck[a]
            rest = without(truck, a)
            b = rand(rng, 0:length(rest))
            t = vcat(rest[1:b], c, rest[b+1:end])
            fixed = repaired(P, t, sorties)
            fixed !== nothing && (s = fixed)
        elseif op == 5 && length(truck) >= 3
            a = rand(rng, 1:length(truck)-1)
            b = rand(rng, a+1:length(truck))
            t = copy(truck)
            reverse!(t, a, b)
            fixed = repaired(P, t, sorties)
            fixed !== nothing && (s = fixed)
        else
            continue
        end
        t == truck && s == sorties && nz == z && continue
        v = evaluate(P, t, s, nz)
        v !== nothing && return t, s, nz, v
    end
    return nothing
end
