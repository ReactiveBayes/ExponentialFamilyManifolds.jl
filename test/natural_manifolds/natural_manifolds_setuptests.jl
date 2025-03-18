
using StableRNGs, ExponentialFamily, Manifolds, ManifoldsBase, LinearAlgebra

import ExponentialFamilyManifolds: get_natural_manifold, partition_point

using FastCholesky

function test_natural_manifold(f; seed=42, ndistributions=100, test_metric=true, test_injectivity_radius=true)
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

        if test_metric && M.metric isa ExponentialFamilyManifolds.FisherInformationMetric
            @test ManifoldsBase.get_basis_default(M, η) isa
                ExponentialFamilyManifolds.NaturalBasis
            @test Manifolds.local_metric(M, η, ManifoldsBase.get_basis_default(M, η)) ≈
                ExponentialFamily.fisherinformation(ef)
            @test Manifolds.inverse_local_metric(M, η, ManifoldsBase.get_basis_default(M, η)) ≈
                cholinv(ExponentialFamily.fisherinformation(ef))
        end
    end

    distribution = f(rng)
    sample = rand(rng, distribution)
    dims = size(sample)
    ef = convert(ExponentialFamilyDistribution, distribution)

    T = ExponentialFamily.exponential_family_typetag(ef)
    M = get_natural_manifold(T, dims, getconditioner(ef))
    if test_metric && M.metric isa ExponentialFamilyManifolds.FisherInformationMetric
        @testset "Testing $(typeof(M)) with retractions" begin
            pts = [rand(rng, M) for _ in 1:5]
            Manifolds.test_manifold(
                M, 
                pts;
                test_exp_log=true,
                default_inverse_retraction_method=nothing,
                test_default_vector_transport=false,
                retraction_methods=[
                    ExponentialFamilyManifolds.FirstOrderRetraction(),
                ],
                inverse_retraction_methods=[],
                test_injectivity_radius=test_injectivity_radius,
            )
        end
    end
end
