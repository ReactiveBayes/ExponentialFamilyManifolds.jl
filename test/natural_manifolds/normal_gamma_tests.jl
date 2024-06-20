@testitem "Check `NormalGamma` natural manifold" begin
    include("natural_manifolds_setuptests.jl")

    test_natural_manifold() do rng
        return NormalGamma(10rand(rng), 10rand(rng), 10rand(rng), 10rand(rng))
    end
end