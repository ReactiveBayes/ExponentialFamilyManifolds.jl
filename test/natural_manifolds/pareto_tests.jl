@testitem "Check `Pareto` natural manifold" begin
    include("natural_manifolds_setuptests.jl")

    test_natural_manifold() do rng
        return Pareto(10rand(rng), 10rand(rng))
    end
end
