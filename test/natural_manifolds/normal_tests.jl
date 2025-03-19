@testitem "Check `Normal` natural manifold" begin
    include("natural_manifolds_setuptests.jl")

    test_natural_manifold() do rng
        return NormalMeanVariance(10randn(rng), 10rand(rng))
    end
end

@testitem "Check `MvNormal` natural manifold" begin
    include("natural_manifolds_setuptests.jl")

    test_natural_manifold() do rng
        k = rand(rng, 1:10)
        m = randn(k)
        L = LowerTriangular(randn(k, k))
        C = L * L' + k * I
        return MvNormalMeanCovariance(m, C)
    end
end

@testitem "Check `MvNormalMeanScalePrecision` natural manifold" begin
    include("natural_manifolds_setuptests.jl")

    test_natural_manifold() do rng
        k = rand(rng, 1:10)
        m = randn(rng, k)
        γ = rand(rng)^2 + 1
        return MvNormalMeanScalePrecision(m, γ)
    end
end

@testitem "Check MLE works for `MvNormalMeanScalePrecision`" begin
    include("natural_manifolds_setuptests.jl")
    using Manopt
    import Distributions: kldivergence, Distribution

    test_mle_works(; mle_samples=1000, ndistributions=3) do rng
        return MvNormalMeanScalePrecision(randn(rng, 2), 2)
    end
end

@testitem "Check MLE works for `NormalMeanVariance`" begin
    include("natural_manifolds_setuptests.jl")
    using Manopt
    import Distributions: kldivergence, Distribution

    test_mle_works(; mle_samples=500, ndistributions=3) do rng
        return NormalMeanVariance(randn(rng), 1 / 2)
    end
end

@testitem "Check MLE works for `MvNormalMeanCovariance`" begin
    include("natural_manifolds_setuptests.jl")
    using Manopt
    import Distributions: kldivergence, Distribution

    @test_broken test_mle_works(;
        mle_samples=500, ndistributions=3, kl_friendly=false
    ) do rng
        k = 2
        m = randn(rng, k)
        L = LowerTriangular(randn(rng, k, k))
        C = L * L' + k * I
        return MvNormalMeanCovariance(m, C)
    end
end
