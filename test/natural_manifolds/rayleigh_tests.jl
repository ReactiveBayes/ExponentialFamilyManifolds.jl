@testitem "Check `Rayleigh` natural manifold" begin
    include("natural_manifolds_setuptests.jl")

    test_natural_manifold() do rng
        return Rayleigh(10rand(rng))
    end
end

@testitem "Check MLE works for `Rayleigh`" begin
    include("mle_manifolds_setuptests.jl")
    test_mle_works(; mle_samples=500, ndistributions=3) do rng
        return Rayleigh(rand(rng))
    end
end
