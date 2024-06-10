@testitem "Check `LogNormal` natural manifold" begin
    
    include("natural_manifolds_setuptests.jl")

    test_natural_manifold() do rng
        return LogNormal(10randn(rng), 10rand(rng))
    end

end