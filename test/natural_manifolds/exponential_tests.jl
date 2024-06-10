@testitem "Check `Exponential` natural manifold" begin
    include("natural_manifolds_setuptests.jl")

    test_natural_manifold() do rng
        return Exponential(100rand(rng))
    end
end