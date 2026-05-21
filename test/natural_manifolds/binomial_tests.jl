@testitem "Check `Binomial` natural manifold" begin
    include("natural_manifolds_setuptests.jl")

    test_natural_manifold() do rng
        return Binomial(rand(rng, 1:20), rand(rng, 0:0.001:1))
    end
end

@testitem "Check MLE works for `Binomial`" begin
    include("mle_manifolds_setuptests.jl")
    # Use fewer samples/iterations for faster tests with explicit conditioner handling
    test_mle_works(; mle_samples=200, ndistributions=3) do rng
        dist = Binomial(rand(rng, 1:20), rand(rng, 0:0.001:1))
        return dist
    end
end

@testitem "Conditioner errors" begin
    import ExponentialFamily: Binomial
    import ExponentialFamilyManifolds: get_natural_manifold_base, partition_point
    @test_throws ArgumentError get_natural_manifold_base(Binomial, ())
    @test_throws ArgumentError get_natural_manifold_base(Binomial, (), -1.0)
    @test_throws ArgumentError partition_point(Binomial, (), 0.5)
    @test_throws ArgumentError partition_point(Binomial, (), 0.5, -1.0)
end
