@testitem "Check `Weibull` natural manifold" begin
    include("natural_manifolds_setuptests.jl")

    test_natural_manifold() do rng
        return Weibull(10rand(rng), 10rand(rng))
    end
end

@testitem "Check MLE works for `Weibull`" begin
    include("natural_manifolds_setuptests.jl")
    using Manopt
    import Distributions: kldivergence, Distribution

    test_mle_works(mle_samples=500, ndistributions=3) do rng
        return Weibull(rand(rng), rand(rng))
    end
end