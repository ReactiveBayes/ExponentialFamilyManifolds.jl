@testitem "Check `Exponential` natural manifold" begin
    include("natural_manifolds_setuptests.jl")

    test_natural_manifold() do rng
        return Exponential(100rand(rng))
    end
end

@testitem "Check MLE works for `Exponential`" begin
    include("mle_manifolds_setuptests.jl")
    test_mle_works(; mle_samples=500, ndistributions=3) do rng
        return Exponential(rand(rng))
    end
end
