# The metaheuristic from the TTP-D paper: a nearest-neighbour start, a descent through ten neighbourhoods,
# then random moves, accepting worse plans less and less often, restarting from the best when it goes cold.

function nearest_route(P::Problem, customers)
    left = collect(customers)
    here = 0
    route = Int[]
    while !isempty(left)
        i = argmin([d(P, here, c) for c in left])
        here = left[i]
        push!(route, here)
        deleteat!(left, i)
    end
    return route
end

function start_heat(P, truck, sorties, z, rng; accept=0.8, samples=80)
    base = evaluate(P, truck, sorties, z)
    worse = Float64[]
    for _ in 1:samples
        out = random_move(P, truck, sorties, z, rng)
        out !== nothing && out[4] < base && push!(worse, base - out[4])
    end
    isempty(worse) ? max(abs(base) * 0.05, 1.0) : max(sum(worse) / length(worse) / -log(accept), 1e-6)
end

function metaheuristic(P::Problem, seconds::Real; seed::Integer=0, cooling=0.97, floor=1e-4)
    rng = MersenneTwister(seed)
    deadline = time() + seconds
    cap = P.n <= 20 ? nothing : 30 * P.n

    truck = nearest_route(P, 1:P.n)
    sorties = Sortie[]
    z = fill(false, P.n + 2)
    value = evaluate(P, truck, sorties, z)
    truck, sorties, z, value = descend(P, truck, sorties, z, value, rng, deadline, cap)
    current = best = (truck, sorties, z, value)

    moves = max(20, 12 * (P.n + 1))
    hot = start_heat(P, truck, sorties, z, rng)
    heat, cold = hot, hot * floor
    while time() < deadline
        for _ in 1:moves
            time() >= deadline && break
            out = random_move(P, current[1], current[2], current[3], rng)
            out === nothing && continue
            change = out[4] - current[4]
            if change >= 0 || rand(rng) < exp(change / heat)
                current = out
                out[4] > best[4] + EPS && (best = out)
            end
        end
        heat *= cooling
        if heat < cold
            heat, current = hot, best
        end
    end

    polished = descend(P, best..., rng, deadline + min(seconds * 0.1, 5.0), cap)
    polished[4] > best[4] + EPS && (best = polished)
    trouble = problem_with(P, best[1], best[2], best[3])
    trouble === nothing || error("the search returned an infeasible plan: $trouble")
    return best
end
