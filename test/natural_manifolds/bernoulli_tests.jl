@testitem "Check `Bernoulli` natural manifold" begin
    include("natural_manifolds_setuptests.jl")

    test_natural_manifold() do rng
        return Bernoulli(rand(rng))
    end
end

@testitem "Check MLE works for `Bernoulli`" begin
    include("mle_manifolds_setuptests.jl")
    test_mle_works() do rng
        return Bernoulli(rand(rng))
    end
end
