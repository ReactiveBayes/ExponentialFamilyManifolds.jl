using StableRNGs, ExponentialFamily, ManifoldsBase, LinearAlgebra
using Distributions

import ExponentialFamilyManifolds: get_fisher_manifold, partition_point

function test_metric_manifold(
    f; seed=42, ndistributions=100, test_points=10, maximal_norm=1, tol=1e-3
)
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

        # check jacobians are inverses
        natural_manifold = ManifoldsBase.decorated_manifold(M)
        for _ in 1:test_points
            p = rand(rng, M)
            v = rand(rng, M; vector_at=p)
            
            v_nat = ExponentialFamilyManifolds.jacobian_manifold_to_nat(natural_manifold, v)
            v_back = ExponentialFamilyManifolds.jacobian_nat_to_manifold(
                natural_manifold, v_nat
            )
            @test v ≈ v_back

            v_nat_2 = rand(rng, M; vector_at=p) # vectors are in same space
            v_man = ExponentialFamilyManifolds.jacobian_nat_to_manifold(
                natural_manifold, v_nat_2
            )
            v_nat_back = ExponentialFamilyManifolds.jacobian_manifold_to_nat(
                natural_manifold, v_man
            )
            @test v_nat_2 ≈ v_nat_back
        end
        
        # Explicit fisher check for one point
        p_test = rand(rng, M)
        X_test = rand(rng, M; vector_at=p_test)
        fisher_matrix = fisherinformation(convert(ExponentialFamilyDistribution, M, p_test))
        
        # Since for Bernoulli and most simple cases without coordinate change, 
        # jacobian is Identity, so inner should be X' * Fisher * X.
        # We check if the inner product computed by the manifold matches explicit Fisher calculation.
        # But first we need to know if there is a coordinate change.
        # For Bernoulli, natural params are just the vector itself, so no coordinate change.
        
        X_nat_test = ExponentialFamilyManifolds.jacobian_manifold_to_nat(natural_manifold, X_test)
        expected_inner = dot(X_nat_test, fisher_matrix, X_nat_test)
        
        @test inner(M, p_test, X_test, X_test) ≈ expected_inner

        #respect fisher metric
        for _ in 1:test_points
            p = rand(rng, M)
            v = rand(rng, M; vector_at=p)
            v_norm = maximal_norm*v ./ norm(M, p, v)
            v2 = rand(rng, M; vector_at=p)
            @test check_geodesic(M, p, v_norm, tol=tol, error=:warn)
            default_vector_transport = ManifoldsBase.default_vector_transport_method(M)
            @test check_vector_transport(M, default_vector_transport, p, v_norm, v2)
        end
    end
end
