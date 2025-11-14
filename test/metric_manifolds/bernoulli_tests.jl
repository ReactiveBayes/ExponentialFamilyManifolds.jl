@testitem "Check `Bernoulli` fisher manifold" begin
    include("metric_manifolds_setuptests.jl")
    test_metric_manifold() do rng
        return Bernoulli(rand(rng))
    end
end
