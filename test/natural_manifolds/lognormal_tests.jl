@testitem "Check `LogNormal` natural manifold" begin
    include("natural_manifolds_setuptests.jl")

    test_natural_manifold() do rng
        return LogNormal(10randn(rng), 10rand(rng))
    end
end

@testitem "Check MLE works for `LogNormal`" begin
    include("natural_manifolds_setuptests.jl")
    using Manopt
    import Distributions: kldivergence, Distribution

    # Use fewer samples/iterations for faster tests
    test_mle_works(mle_samples=500, ndistributions=3) do rng
        return LogNormal(randn(rng), rand(rng))
    end
end