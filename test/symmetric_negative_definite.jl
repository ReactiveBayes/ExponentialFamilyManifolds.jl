@testitem "Generic properties of Negative Definite Matrices" begin
    import ExponentialFamilyManifolds: SymmetricNegativeDefinite
    import ManifoldsBase: check_point, check_vector
    using ManifoldsBase, StaticArrays, JET, Manifolds, StableRNGs, LinearAlgebra

    rng = StableRNG(42)

    for size in (2, 4, 8)
        M = SymmetricNegativeDefinite(size)

        @test repr(M) == "SymmetricNegativeDefinite($size)"

        @test @inferred(representation_size(M)) === (size, size)
        @test @inferred(manifold_dimension(M)) === ((size) * (size + 1)) ÷ 2
        @test @inferred(injectivity_radius(M)) === Inf
        @test @inferred(is_flat(M)) === false
        @test @inferred(get_embedding(M)) === Euclidean(size, size; field=ℝ)

        @test @allocated(representation_size(M)) === 0
        @test @allocated(manifold_dimension(M)) === 0
        @test @allocated(injectivity_radius(M)) === 0
        @test @allocated(is_flat(M)) === 0
        @test @allocated(get_embedding(M)) === 0

        @test_opt representation_size(M)
        @test_opt manifold_dimension(M)
        @test_opt injectivity_radius(M)
        @test_opt is_flat(M)
        @test_opt get_embedding(M)

        p = rand(rng, M)

        @test isposdef(-p)

        p = rand(rng, size, size)
        @test !isposdef(p)
        @test !isposdef(-p)
        rand!(rng, M, p)
        @test isposdef(-p)

        p = rand(rng, M)
        @test check_point(M, p) === nothing
        @test check_point(M, -p) isa DomainError

        @test_opt check_point(M, p)

        X = Symmetric(rand(rng, size, size))
        @test check_vector(M, p, X) === nothing
        @test check_vector(M, p, -X) === nothing
        X = rand(rng, size, size)
        @test check_vector(M, p, X) isa DomainError
        @test check_vector(M, p, -X) isa DomainError

        @test_opt check_vector(M, p, X)
    end
end

@testitem "Manifolds.test_manifold" begin
    using Manifolds, Static, Random, StaticArrays, StableRNGs, LinearAlgebra

    import ExponentialFamilyManifolds: SymmetricNegativeDefinite

    rng = StableRNG(42)

    for size in 2:5
        M = SymmetricNegativeDefinite(size)

        ptss = [
            [
                -10 * Matrix(Diagonal(ones(size))),
                -20 * Matrix(Diagonal(ones(size))),
                -30 * Matrix(Diagonal(ones(size))),
            ],
            [
                -1 * Matrix(Diagonal(ones(size))),
                -2 * Matrix(Diagonal(ones(size))),
                -3 * Matrix(Diagonal(ones(size))),
            ],
        ]

        # An example from Manifolds.jl
        if size === 3
            A(α) = [1.0 0.0 0.0; 0.0 cos(α) sin(α); 0.0 -sin(α) cos(α)]
            ptsF = [#
                -1 * [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1],
                -1 * [2.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 1],
                (
                    A(π / 6) *
                    (-1 * [1.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 1.0]) *
                    transpose(A(π / 6))
                ),
            ]
            push!(ptss, ptsF)
        end

        for pts in ptss
            Manifolds.test_manifold(
                M,
                pts;
                test_vector_spaces=true,
                test_rand_point=true,
                test_rand_tvector=true,
                test_inplace=true,
                test_is_tangent=true,
                test_mutating_rand=true,
                test_project_tangent=true,
                test_default_vector_transport=true,
                is_tangent_atol_multiplier=1e1,
                exp_log_atol_multiplier=1e2,
                # vector_transport_methods = [ParallelTransport()],
            )
        end
    end
end

@testitem "negate!" begin
    import ExponentialFamilyManifolds: negate!
    using JET

    for n in (2:10)
        M = rand(n, n)
        C = copy(M)
        @test all(@inferred(negate!(M)) .≈ -C)
        @test all(@inferred(negate!(M)) .≈ C)
        @test @allocated(negate!(M)) === 0
        @test_opt negate!(M)
    end
end

@testitem "Negated" begin
    import ExponentialFamilyManifolds: negate!
    import ExponentialFamilyManifolds: SymmetricNegativeDefinite, Negated
    using JET, Manifolds, LinearAlgebra

    m = rand(10, 10)
    mn = Negated(m)

    @test repr(mn) == "Negated($m)"

    for i in eachindex(m)
        @test m[i] ≈ -mn[i]
    end

    for i in eachindex(mn)
        @test mn[i] ≈ -m[i]
    end

    for (a, b) in zip(m, mn)
        @test a ≈ -b
    end

    @test collect(m) ≈ -collect(mn)
    @test eltype(m) === eltype(mn)
    @test size(m) === size(mn)
    @test length(m) === length(mn)
    for i in 1:10, j in 1:10
        @test getindex(m, i, j) ≈ -getindex(mn, i, j)
        @test @eval(@allocated(getindex($mn, $i, $j))) === 0
    end

    @test_opt getindex(mn, 1, 1)
    @test_opt collect(mn)
    @test_opt eltype(mn)
    @test_opt size(mn)
    @test_opt length(mn)

    @inferred getindex(mn, 1, 1)
    @inferred collect(mn)
    @inferred eltype(mn)
    @inferred size(mn)
    @inferred length(mn)

    @testset for n in (2:10)
        m = rand(SymmetricPositiveDefinite(n))
        @test isposdef(m)
        @test !@inferred(isposdef(Negated(m)))

        m = rand(SymmetricNegativeDefinite(n))
        @test !isposdef(m)
        @test @inferred(isposdef(Negated(m)))

        @test_opt isposdef(Negated(m))
    end
end

@testitem "Simple manifolds optimization #1" begin
    import ExponentialFamilyManifolds: SymmetricNegativeDefinite
    using ForwardDiff, StableRNGs, LinearAlgebra, Manopt, ManifoldsBase

    a = 1
    b = -10
    c = 3

    f(M, X) = a * norm(X)^2 + b * norm(X) + c
    g(M, X) = ForwardDiff.gradient((x) -> f(M, x), X)

    for size in (3, 5, 7), eps in (1e-3, 1e-5), stepsize in (0.1, 0.01, 0.001)
        M = SymmetricNegativeDefinite(size)
        p0 = rand(StableRNG(42), M)

        expected_n = -b / 2a
        expected_minimum = c - b^2 / (4a)

        q1 = gradient_descent(
            M,
            f,
            g,
            p0;
            stepsize=ConstantStepsize(stepsize),
            stopping_criterion=StopWhenGradientNormLess(eps),
        )

        @test all(<(0), eigvals(q1))
        @test norm(q1) ≈ expected_n rtol = eps
        @test f(M, q1) ≈ expected_minimum rtol = 1e-7
        @test norm(M, q1, g(M, q1)) <= eps
    end
end