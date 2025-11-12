@testitem "Check `Dirichlet` natural manifold" begin
    include("natural_manifolds_setuptests.jl")

    test_natural_manifold() do rng
        k = rand(rng, 2:10)
        return Dirichlet(10rand(rng, k))
    end
end

@testitem "Check MLE works for `Dirichlet`" begin
    include("mle_manifolds_setuptests.jl")
    # Use fewer samples/iterations for faster tests
    test_mle_works(; mle_samples=500, ndistributions=3) do rng
        return Dirichlet(10rand(rng, 3))
    end
end

@testitem "Dirichlet dimension errors" begin
    import ExponentialFamily: Dirichlet
    import ExponentialFamilyManifolds: get_natural_manifold, get_natural_manifold_base
    @test_throws ArgumentError get_natural_manifold(Dirichlet, ())
    @test_throws ArgumentError get_natural_manifold_base(Dirichlet, ())
end
