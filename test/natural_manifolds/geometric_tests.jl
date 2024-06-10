@testitem "Check `Geometric` natural manifold" begin
    
    include("natural_manifolds_setuptests.jl")

    test_natural_manifold() do rng
        return Geometric(rand(rng))
    end

end