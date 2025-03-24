using StableRNGs, ExponentialFamily, ManifoldsBase, LinearAlgebra
using Distributions

import ExponentialFamilyManifolds: get_natural_manifold, partition_point

function test_mle_works(f; seed=42, mle_samples=1000, ndistributions=10, kl_friendly=true)
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

        all_stats = map(samples) do s
            ExponentialFamily.pack_parameters(ExponentialFamily.sufficientstatistics(ef, s))
        end
        mean_stats = mean(all_stats)

        function cost(M, p)
            ef_candidate = convert(ExponentialFamilyDistribution, M, p)
            return -mean(s -> ExponentialFamily.logpdf(ef_candidate, s), samples)
        end

        function grad(M, p)
            result_container = similar(p)

            ef_candidate = convert(ExponentialFamilyDistribution, M, p)

            grad_A = gradlogpartition(ef_candidate)

            if grad_A isa Tuple && mean_stats isa Tuple
                raw_grad = ntuple(i -> grad_A[i] - mean_stats[i], length(grad_A))
            else
                raw_grad = grad_A .- mean_stats
            end

            result_container .= raw_grad

            return result_container
        end

        function manopt_cost(M, p)
            return cost(M, p)
        end

        function manopt_grad(M, p)
            g = grad(M, p)
            return ManifoldsBase.project(M, p, g)
        end

        p_mle = gradient_descent(M, manopt_cost, manopt_grad, rand(rng, M))
        ef_mle = convert(ExponentialFamilyDistribution, M, p_mle)
        if kl_friendly
            kl_div = kldivergence(convert(Distribution, ef_mle), distribution)
            @test kl_div < 0.1
        else
            @test getnaturalparameters(ef_mle) ≈ getnaturalparameters(ef) atol = 9e-1
        end
    end
end

function test_natural_manifold(f; seed=42, ndistributions=100)
    rng = StableRNG(seed)

    foreach(1:ndistributions) do _
        distribution = f(rng)
        sample = rand(rng, distribution)
        dims = size(sample)

        ef = convert(ExponentialFamilyDistribution, distribution)
        T = ExponentialFamily.exponential_family_typetag(ef)
        M = get_natural_manifold(T, dims, getconditioner(ef))
        η = partition_point(T, dims, getnaturalparameters(ef), getconditioner(ef))

        @test is_point(M, η, error=:error)
        ef_back = convert(ExponentialFamilyDistribution, M, η)
        @test getnaturalparameters(ef_back) ≈ getnaturalparameters(ef)
        @test getconditioner(ef_back) == getconditioner(ef)
        @test isproper(ef_back) == true
    end
end
