@testitem "Check `NegativeBinomial` natural manifold" begin
    include("natural_manifolds_setuptests.jl")

    test_natural_manifold() do rng
        return NegativeBinomial(10rand(rng), rand(rng, 0.00001:0.01:1))
    end
end

@testitem "Check MLE works for `NegativeBinomial`" begin
    include("natural_manifolds_setuptests.jl")
    using Manopt
    import Distributions: kldivergence, Distribution

    @test_broken test_mle_works(mle_samples=1000, ndistributions=3) do rng
        return NegativeBinomial(rand(rng), rand(rng, 0.00001:0.01:1))
    end
end