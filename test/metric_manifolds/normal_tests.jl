@testitem "Check `Normal` natural manifold" begin
    include("metric_manifolds_setuptests.jl")

    test_metric_manifold(tol=1e-5, maximal_norm=0.3 ) do rng
        return NormalMeanVariance(10randn(rng), rand(rng)+1)
    end
end

@testitem "Check `Normal` fisher manifold MLE works" begin
    include("mle_metric_manifolds_setuptests.jl")
    test_mle_works() do rng
        return NormalMeanVariance(10randn(rng), 10rand(rng))
    end
end