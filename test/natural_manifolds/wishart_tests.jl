@testitem "Check `Wishart` natural manifold" begin
    include("natural_manifolds_setuptests.jl")
    import ExponentialFamily: WishartFast
    test_natural_manifold() do rng
        k = rand(rng, 2:10)
        L = LowerTriangular(randn(rng, k, k))
        C = L * L' + k * I
        return WishartFast(k + 2, C)
    end
end

@testitem "Check MLE works for `Wishart`" begin
    include("natural_manifolds_setuptests.jl")

    using Manopt
    import Distributions: kldivergence, Distribution
    import ExponentialFamily: WishartFast

    test_mle_works(mle_samples=500, ndistributions=1, kl_friendly=false) do rng
        k = 2
        L = LowerTriangular(randn(rng, k, k))
        C = L * L' + k * I
        return WishartFast(k + 2, C)
    end
end