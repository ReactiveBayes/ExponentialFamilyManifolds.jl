@testitem "Check `Bernoulli` natural manifold" begin
    
    include("natural_manifolds_setuptests.jl")

    test_natural_manifold() do rng
        return Bernoulli(rand(rng))
    end

end