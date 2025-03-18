@testitem "Natural manifold properties" begin
    using Distributions, ExponentialFamily, ManifoldsBase, StableRNGs, JET, LinearAlgebra

    import ExponentialFamilyManifolds:
        NaturalParametersManifold,
        ShiftedPositiveNumbers,
        get_natural_manifold,
        get_natural_manifold_base,
        partition_point,
        getbase

    @testset "Without dimension" begin
        M = @inferred(get_natural_manifold(Beta, ()))

        @test M isa NaturalParametersManifold
        @test M isa AbstractManifold
        @test getbase(M) === get_natural_manifold_base(Beta, ())

        @test_opt get_natural_manifold(Beta, ())

        # Generate a couple of random Beta distributions 
        # and check that their natural parameters are within the 
        # natural parameters manifold
        for _ in 1:10
            dist = Beta(rand() + 1, rand() + 10)
            ef = convert(ExponentialFamilyDistribution, dist)
            η = getnaturalparameters(ef)

            @test @inferred(
                getnaturalparameters(
                    convert(ExponentialFamilyDistribution, M, partition_point(M, η))
                )
            ) == η

            p = partition_point(M, η)
            pr = rand(StableRNG(42), M)
            X = ManifoldsBase.zero_vector(M, p)

            @test ManifoldsBase.is_point(M, p)
            @test ManifoldsBase.is_point(M, pr)
            @test ManifoldsBase.is_vector(M, p, X)

            @test_opt partition_point(M, η)
        end
    end

    @testset "With dimension" begin
        for dim in 3:5
            M = get_natural_manifold(MvNormalMeanCovariance, (dim,))

            @test M isa NaturalParametersManifold
            @test M isa AbstractManifold
            @test getbase(M) === get_natural_manifold_base(MvNormalMeanCovariance, (dim,))

            for _ in 1:10
                dist = MvNormalMeanCovariance(ones(dim), Matrix(Diagonal(ones(dim))))
                ef = convert(ExponentialFamilyDistribution, dist)
                η = getnaturalparameters(ef)

                @test @inferred(
                    getnaturalparameters(
                        convert(ExponentialFamilyDistribution, M, partition_point(M, η))
                    )
                ) == η

                p = partition_point(M, η)
                pr = rand(StableRNG(42), M)
                X = ManifoldsBase.zero_vector(M, p)

                @test ManifoldsBase.is_point(M, p, error=:error)
                @test ManifoldsBase.is_point(M, pr)
                @test ManifoldsBase.is_vector(M, p, X)

                @test_opt partition_point(M, η)
            end
        end
    end
end

@testitem "Natural manifold properties: BaseMetric" begin
    using Distributions, ExponentialFamily, Manifolds, ManifoldsBase, StableRNGs, JET, LinearAlgebra

    import ExponentialFamilyManifolds: BaseMetric

    M = ExponentialFamilyManifolds.get_natural_manifold(Beta, (), nothing, BaseMetric())
    @test M.metric isa BaseMetric
    @test ExponentialFamilyManifolds.select_skip_methods(ManifoldsBase.retract, M) == ManifoldsBase.IsExplicitDecorator()
    p = rand(StableRNG(42), M)
    q = copy(p)
    @test  ManifoldsBase.retract!(M, q, p, p) ≈ ManifoldsBase.retract!(M.base, q, p, p)
end

