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
