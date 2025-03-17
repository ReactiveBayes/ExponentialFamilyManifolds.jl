
using StableRNGs, ExponentialFamily, Manifolds, ManifoldsBase, LinearAlgebra

import ExponentialFamilyManifolds: get_natural_manifold, partition_point

function test_natural_manifold(f; seed=42, ndistributions=100, test_metric=true)
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

        if test_metric
            @test M.metric isa ExponentialFamilyManifolds.FisherInformationMetric
            @test ManifoldsBase.get_basis_default(M, η) isa ExponentialFamilyManifolds.NaturalBasis
            @test Manifolds.local_metric(M, η, ManifoldsBase.get_basis_default(M, η)) ≈ ExponentialFamily.fisherinformation(ef)
        end
    end
end
