@testitem "Check `Chisq` natural manifold" begin
    include("natural_manifolds_setuptests.jl")

    test_natural_manifold() do rng
        return Chisq(10rand(rng))
    end
end