@testitem "Check `Binomial` natural manifold" begin
    include("natural_manifolds_setuptests.jl")

    test_natural_manifold() do rng
        return Binomial(rand(rng, 1:20), rand(rng, 0:0.001:1))
    end
end
