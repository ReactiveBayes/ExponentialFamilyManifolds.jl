@testitem "Check `Gamma` natural manifold" begin
    include("natural_manifolds_setuptests.jl")

    test_natural_manifold() do rng
        return Gamma(10rand(rng), 10rand(rng))
    end
end

@testitem "Check MLE works for `Gamma`" begin
    include("natural_manifolds_setuptests.jl")
    using Manopt
    import Distributions: kldivergence, Distribution

    @test_broken test_mle_works(; mle_samples=500, ndistributions=3) do rng
        return Gamma(rand(rng), 10rand(rng))
    end
end
