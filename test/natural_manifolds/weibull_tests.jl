@testitem "Check `Weibull` natural manifold" begin
    include("natural_manifolds_setuptests.jl")

    test_natural_manifold() do rng
        return Weibull(10rand(rng), 10rand(rng))
    end
end