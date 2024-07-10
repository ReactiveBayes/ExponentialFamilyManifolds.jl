@testitem "Check `Categorical` natural manifold" begin
    include("natural_manifolds_setuptests.jl")
    
    test_natural_manifold() do rng
        p = rand(rng, 10)
        normalize!(p, 1)
        return Categorical(p)
    end
end