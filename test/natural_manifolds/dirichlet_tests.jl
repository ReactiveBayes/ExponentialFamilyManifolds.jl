@testitem "Check `Dirichlet` natural manifold" begin
    include("natural_manifolds_setuptests.jl")

    test_natural_manifold() do rng
        k = rand(rng, 2:10)
        return Dirichlet(10rand(rng, k))
    end
end