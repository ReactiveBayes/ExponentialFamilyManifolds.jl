
using StableRNGs, ExponentialFamily, ManifoldsBase, LinearAlgebra

import ExponentialFamilyManifolds: get_natural_manifold, partition_point

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