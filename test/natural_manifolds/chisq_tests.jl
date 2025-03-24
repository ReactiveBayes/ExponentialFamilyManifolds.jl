@testitem "Check `Chisq` natural manifold" begin
    include("natural_manifolds_setuptests.jl")

    test_natural_manifold() do rng
        return Chisq(10rand(rng))
    end
end

@testitem "Check MLE works for `Chisq`" begin
    include("mle_manifolds_setuptests.jl")
    # Use fewer samples/iterations for faster tests
    test_mle_works(; mle_samples=200, ndistributions=3) do rng
        return Chisq(10rand(rng))
    end
end
