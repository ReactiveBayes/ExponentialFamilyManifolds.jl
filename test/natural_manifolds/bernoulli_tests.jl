@testitem "Check `Bernoulli` natural manifold" begin
    include("natural_manifolds_setuptests.jl")

    test_natural_manifold() do rng
        return Bernoulli(rand(rng))
    end
end

@testitem "Check MLE works for `Bernoulli`" begin
    include("natural_manifolds_setuptests.jl")

    using Manopt
    import Distributions: kldivergence, Distribution

    test_mle_works(mle_samples=1000, ndistributions=3) do rng
        return Bernoulli(rand(rng))
    end
end
