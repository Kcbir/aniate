using Aniate, Test

const NODES = [(0, 0), (4, 1), (5, 4), (1, 5), (2, 2), (3, 3)]
const PROFITS, WEIGHTS = [50, 60, 70, 80, 40], [2, 3, 4, 5, 2]
quick = (; seconds=0.3)

@testset "what you leave out decides the problem" begin
    for (plan, name) in ((cube(NODES; quick...), "TSP"), (cube(NODES; drone=true, quick...), "TSP-D"),
                         (cube(NODES, PROFITS, WEIGHTS; quick...), "TTP"),
                         (cube(NODES, PROFITS, WEIGHTS; drone=true, quick...), "TTP-D"))
        @test plan.problem == name
        @test plan.route[1] == plan.route[end] == 0
        @test sort!(vcat(plan.route[2:end-1], [f[2] for f in plan.flights])) == 1:5
        @test plan.time > 0
    end
end

@testset "rules" begin
    @test isempty(cube(NODES, PROFITS, WEIGHTS; quick...).flights)
    @test isempty(cube(NODES; drone=true, quick...).items)
    @test sum(WEIGHTS[i] for i in cube(NODES, PROFITS, WEIGHTS; drone=true, capacity=4, quick...).items; init=0) <= 4
    @test cube(NODES; drone=true, quick...).time <= cube(NODES; quick...).time + 1e-9
    for (a, b, c) in cube(NODES; endurance=6, quick...).flights
        p = [NODES[a+1], NODES[b+1], NODES[c+1]]
        @test hypot((p[1] .- p[2])...) + hypot((p[2] .- p[3])...) <= 6 + 1e-9
    end
    @test cube([(0, 0), (9, 9), (5, 1)]; dist=[0 1 1; 1 0 1; 1 1 0], quick...).time ≈ 3
end

@testset "bad input is named" begin
    @test_throws ArgumentError cube([(0, 0)])
    @test_throws ArgumentError cube(NODES, PROFITS)
    @test_throws ArgumentError cube(NODES, [1], [1])
    @test_throws ArgumentError cube(NODES; endurance=-1)
    @test_throws ArgumentError cube(NODES, PROFITS, WEIGHTS; capacity=0)
    @test_throws ArgumentError cube(NODES; seconds=0)
end
