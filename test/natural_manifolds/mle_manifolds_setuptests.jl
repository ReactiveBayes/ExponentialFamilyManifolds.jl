using StableRNGs, ExponentialFamily, ManifoldsBase, LinearAlgebra
using Distributions

using Manopt
import Distributions: kldivergence, Distribution

import ExponentialFamilyManifolds: get_natural_manifold, partition_point
import ADTypes: AutoForwardDiff
using ManifoldDiff
import ManifoldDiff: TangentDiffBackend

function test_mle_works(
    f;
    seed=42,
    mle_samples=1000,
    ndistributions=10,
    backend_type=AutoForwardDiff(),
    kl_friendly=true,
)
    rng = StableRNG(seed)

    foreach(1:ndistributions) do _
        distribution = f(rng)
        ef = convert(ExponentialFamilyDistribution, distribution)
        T = ExponentialFamily.exponential_family_typetag(ef)
        dims = size(rand(rng, distribution))
        conditioner = getconditioner(ef)
        M = get_natural_manifold(T, dims, conditioner)

        # Generate samples from the distribution
        samples = [rand(rng, distribution) for _ in 1:mle_samples]

        function cost(M, p)
            ef_candidate = convert(ExponentialFamilyDistribution, M, p)
            return -mean(s -> ExponentialFamily.logpdf(ef_candidate, s), samples)
        end

        function grad(M, p, backend=TangentDiffBackend(backend_type))
            return ManifoldDiff.gradient(M, (p) -> cost(M, p), p, backend)
        end

        stepsize = DistanceOverGradients()
        p_mle = gradient_descent(M, cost, grad, rand(rng, M); stepsize=stepsize)
        ef_mle = convert(ExponentialFamilyDistribution, M, p_mle)
        if kl_friendly
            kl_div = kldivergence(convert(Distribution, ef_mle), distribution)
            @test kl_div < 0.1
        else
            @test getnaturalparameters(ef_mle) ≈ getnaturalparameters(ef) atol = 4e-1
        end
    end
end
