@testitem "Check `Rayleigh` natural manifold" begin
    include("natural_manifolds_setuptests.jl")

    test_natural_manifold() do rng
        return Rayleigh(10rand(rng))
    end
end
