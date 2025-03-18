@testitem "Check `Binomial` natural manifold" begin
    include("natural_manifolds_setuptests.jl")

    test_natural_manifold() do rng
        return Binomial(rand(rng, 1:20), rand(rng, 0:0.001:1))
    end
end

@testitem "Check MLE works for `Binomial`" begin
    include("natural_manifolds_setuptests.jl")
    using Manopt
    import Distributions: kldivergence, Distribution

    # Use fewer samples/iterations for faster tests with explicit conditioner handling
    test_mle_works(mle_samples=200, ndistributions=3) do rng
        dist = Binomial(rand(rng, 1:20), rand(rng, 0:0.001:1))
        return dist 
    end
end