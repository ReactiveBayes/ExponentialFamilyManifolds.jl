using Test
using Pkg

Pkg.activate(".")


include("metric_manifolds_setuptests.jl")

test_metric_manifold() do rng
    return Bernoulli(rand(rng))
end

M = ExponentialFamilyManifolds.get_natural_manifold(Bernoulli, ())
FM = ExponentialFamilyManifolds.with_natural_metric(M)
p = rand(FM)
right_2 = ExponentialFamilyManifolds._geodesic_eta(p, [1.0])
@test exp(FM, p, [1]) == ExponentialFamilyManifolds._geodesic_eta(p, [1.0])