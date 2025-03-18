@testitem "Check `Beta` natural manifold" begin
    include("natural_manifolds_setuptests.jl")

    test_natural_manifold() do rng
        return Beta(10rand(rng), 10rand(rng))
    end
end

@testitem "Check MLE works for `Beta`" begin
    include("natural_manifolds_setuptests.jl")
    using Manopt
    import Distributions: kldivergence, Distribution

    # Use fewer samples/iterations for faster tests
    test_mle_works(mle_samples=1000, ndistributions=3) do rng
        return Beta(10rand(rng), 10rand(rng))
    end
end