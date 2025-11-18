using StableRNGs, ExponentialFamily, ManifoldsBase, LinearAlgebra
using Distributions

import ExponentialFamilyManifolds:
    get_fisher_manifold, partition_point

function test_metric_manifold(f; seed=42, ndistributions=100, test_points=10, maximal_norm=1, tol=1e-3)
    rng = StableRNG(seed)

    foreach(1:ndistributions) do _
        distribution = f(rng)
        sample = rand(rng, distribution)
        dims = size(sample)

        ef = convert(ExponentialFamilyDistribution, distribution)
        T = ExponentialFamily.exponential_family_typetag(ef)
        M = get_fisher_manifold(T, dims, getconditioner(ef))
        η = partition_point(T, dims, getnaturalparameters(ef), getconditioner(ef))

        @test is_point(M, η, error=:error)
        ef_back = convert(ExponentialFamilyDistribution, M, η)
        @test getnaturalparameters(ef_back) ≈ getnaturalparameters(ef)
        @test getconditioner(ef_back) == getconditioner(ef)
        @test isproper(ef_back) == true
        @test ExponentialFamily.exponential_family_typetag(M) ==
            ExponentialFamily.exponential_family_typetag(ef)

        # proper forwarding
        @test ManifoldsBase.get_forwarding_type(M, inner) ==
            ManifoldsBase.StopForwardingType()
        @test ManifoldsBase.get_forwarding_type(M, norm) ==
            ManifoldsBase.StopForwardingType()

        #respect fisher metric
        for _ in 1:test_points
            p = rand(rng, M)
            v = rand(rng, M; vector_at=p)
            v_norm = maximal_norm*v ./ norm(M, p, v)
            @test check_geodesic(M, p, v_norm, tol=tol, error=:warn)
        end
    end
end
