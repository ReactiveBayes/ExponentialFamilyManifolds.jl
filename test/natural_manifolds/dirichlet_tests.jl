@testitem "Check `Dirichlet` natural manifold" begin
    include("natural_manifolds_setuptests.jl")

    test_natural_manifold() do rng
        k = rand(rng, 2:10)
        return Dirichlet(10rand(rng, k))
    end
end

@testitem "Check MLE works for `Dirichlet`" begin
    include("natural_manifolds_setuptests.jl")
    using Manopt
    import Distributions: kldivergence, Distribution

    # Use fewer samples/iterations for faster tests
    test_mle_works(mle_samples=500, ndistributions=3) do rng
        return Dirichlet(10rand(rng, 3))
    end
end