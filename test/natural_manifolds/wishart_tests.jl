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
    include("mle_manifolds_setuptests.jl")

    import ExponentialFamily: WishartFast
    using DifferentiationInterface
    using FiniteDifferences

    test_mle_works(;
        mle_samples=500,
        ndistributions=2,
        backend_type=AutoFiniteDifferences(central_fdm(5, 1)),
        kl_friendly=false,
    ) do rng
        k = 2
        L = LowerTriangular(randn(rng, k, k))
        C = L * L' + k * I
        return WishartFast(3 + rand(rng), C)
    end
end

@testitem "WishartFast dimension errors" begin
    import ExponentialFamily: WishartFast
    import ExponentialFamilyManifolds: get_natural_manifold, get_natural_manifold_base

    @test_throws ArgumentError get_natural_manifold(WishartFast, ())
    @test_throws ArgumentError get_natural_manifold_base(WishartFast, ())
end
