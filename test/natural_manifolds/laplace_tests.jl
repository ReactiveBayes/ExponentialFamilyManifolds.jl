@testitem "Check `Laplace` natural manifold" begin
    include("natural_manifolds_setuptests.jl")

    test_natural_manifold() do rng
        return Laplace(10rand(rng), 10rand(rng))
    end
end

@testitem "Check MLE works for `Laplace`" begin
    include("mle_manifolds_setuptests.jl")

    test_mle_works(; mle_samples=500, ndistributions=3) do rng
        return Laplace(rand(rng), rand(rng))
    end
end
