@testitem "Check `Laplace` natural manifold" begin
    include("natural_manifolds_setuptests.jl")

    test_natural_manifold() do rng
        return Laplace(10rand(rng), 10rand(rng))
    end
end