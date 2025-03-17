@testitem "Check `Poisson` natural manifold" begin
    include("natural_manifolds_setuptests.jl")

    test_natural_manifold() do rng
        return Poisson(rand(rng))
    end
end
