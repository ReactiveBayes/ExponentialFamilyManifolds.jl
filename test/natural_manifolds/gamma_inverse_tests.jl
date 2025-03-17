@testitem "Check `GammaInverse` natural manifold" begin
    include("natural_manifolds_setuptests.jl")

    test_natural_manifold() do rng
        return ExponentialFamily.InverseGamma(10rand(rng), 10rand(rng))
    end
end
