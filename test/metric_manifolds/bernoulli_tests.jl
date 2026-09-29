@testitem "Check `Bernoulli` fisher manifold" begin
    include("metric_manifolds_setuptests.jl")
    test_metric_manifold(maximal_norm=0.3) do rng
        return Bernoulli(rand(rng))
    end
end

@testitem "Check `Bernoulli` fisher manifold MLE works" begin
    include("mle_metric_manifolds_setuptests.jl")
    test_mle_works() do rng
        return Bernoulli(rand(rng))
    end
end
