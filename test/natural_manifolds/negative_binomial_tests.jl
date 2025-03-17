@testitem "Check `NegativeBinomial` natural manifold" begin
    include("natural_manifolds_setuptests.jl")

    test_natural_manifold() do rng
        return NegativeBinomial(10rand(rng), rand(rng, 0.00001:0.01:1))
    end
end
