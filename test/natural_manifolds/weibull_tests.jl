@testitem "Check `Weibull` natural manifold" begin
    include("natural_manifolds_setuptests.jl")

    test_natural_manifold() do rng
        return Weibull(10rand(rng), 10rand(rng))
    end
end

@testitem "Check MLE works for `Weibull`" begin
    include("mle_manifolds_setuptests.jl")
    test_mle_works(; mle_samples=500, ndistributions=3) do rng
        return Weibull(rand(rng), rand(rng))
    end
end

@testitem "Conditioner errors" begin
    import ExponentialFamily: Weibull
    import ExponentialFamilyManifolds: get_natural_manifold_base, partition_point
    @test_throws ArgumentError get_natural_manifold_base(Weibull, ())
    @test_throws ArgumentError get_natural_manifold_base(Weibull, (), -1.0)
    @test_throws ArgumentError partition_point(Weibull, (), 0.5)
    @test_throws ArgumentError partition_point(Weibull, (), 0.5, -1.0)
end
