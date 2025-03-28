@testitem "Check `Poisson` natural manifold" begin
    include("natural_manifolds_setuptests.jl")

    test_natural_manifold() do rng
        return Poisson(rand(rng))
    end
end

@testitem "Check MLE works for `Poisson`" begin
    include("mle_manifolds_setuptests.jl")
    # Use fewer samples/iterations for faster tests
    test_mle_works(; mle_samples=500, ndistributions=3) do rng
        return Poisson(rand(rng))
    end
end
