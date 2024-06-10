@testitem "Natural manifold properties" begin
    using Distributions, ExponentialFamily, ManifoldsBase, StableRNGs, JET, LinearAlgebra

    import ExponentialFamilyManifolds:
        NaturalParametersManifold,
        ShiftedPositiveNumbers,
        get_natural_manifold,
        partition_point

    @testset "Without dimension" begin
        M = @inferred(get_natural_manifold(Beta, ()))

        @test M isa NaturalParametersManifold
        @test M isa AbstractManifold

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
                    convert(ExponentialFamilyDistribution, M, partition_point(M, (), η))
                )
            ) == η

            p = partition_point(M, (), η)
            pr = rand(StableRNG(42), M)
            X = ManifoldsBase.zero_vector(M, p)

            @test ManifoldsBase.is_point(M, p)
            @test ManifoldsBase.is_point(M, pr)
            @test ManifoldsBase.is_vector(M, p, X)

            @test_opt partition_point(M, (), η)
        end
    end

    @testset "With dimension" begin
        for dim in 3:5
            M = get_natural_manifold(MvNormalMeanCovariance, (dim,))

            @test M isa NaturalParametersManifold
            @test M isa AbstractManifold

            for _ in 1:10
                dist = MvNormalMeanCovariance(ones(dim), Matrix(Diagonal(ones(dim))))
                ef = convert(ExponentialFamilyDistribution, dist)
                η = getnaturalparameters(ef)

                @test @inferred(
                    getnaturalparameters(
                        convert(
                            ExponentialFamilyDistribution, M, partition_point(M, (dim,), η)
                        ),
                    )
                ) == η

                p = partition_point(M, (dim,), η)
                pr = rand(StableRNG(42), M)
                X = ManifoldsBase.zero_vector(M, p)

                @test ManifoldsBase.is_point(M, p, error=:error)
                @test ManifoldsBase.is_point(M, pr)
                @test ManifoldsBase.is_vector(M, p, X)

                @test_opt partition_point(M, (dim,), η)
            end
        end
    end
end