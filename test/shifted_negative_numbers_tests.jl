@testitem "Generic properties of Shifted Negative Numbers manifold" begin
    import ExponentialFamilyManifolds:
        ShiftedNegativeNumbers, getshift, shift, unshift, shift!, unshift!
    import ManifoldsBase: check_point, check_vector
    using ManifoldsBase, Static, StaticArrays, JET, Manifolds

    shifts = [0, 0.0, 0.0f0, 1, 1.0, 1.0f0, static(-1), static(1), rand(), randn()]

    for s in shifts
        M = ShiftedNegativeNumbers(s)

        @test repr(M) == "ShiftedNegativeNumbers($s)"

        # We treat this manifold as 1-dimensional vector for convenience
        @test @inferred(representation_size(M)) === (1,)
        @test @inferred(manifold_dimension(M)) === 1
        @test @inferred(injectivity_radius(M)) === Inf
        @test @inferred(is_flat(M)) === false
        @test @inferred(get_embedding(M)) === Euclidean(1; field=ℝ)

        @test_opt representation_size(M)
        @test_opt manifold_dimension(M)
        @test_opt injectivity_radius(M)
        @test_opt is_flat(M)
        @test_opt get_embedding(M)

        @test @inferred(getshift(M)) === s
        @test @inferred(unshift(M, [s])) ≈ [zero(s)]
        @test @inferred(unshift(M, [s + 1])) ≈ [one(s)]
        @test @inferred(unshift(M, s)) ≈ zero(s)
        @test @inferred(unshift(M, s + 1)) ≈ one(s)
        @test @inferred(unshift!(M, [NaN], [s])) ≈ [zero(s)]
        @test @inferred(unshift!(M, [NaN], [s + 1])) ≈ [one(s)]
        @test @inferred(shift(M, [zero(s)])) ≈ [s]
        @test @inferred(shift(M, [one(s)])) ≈ [s + 1]
        @test @inferred(shift(M, zero(s))) ≈ s
        @test @inferred(shift(M, one(s))) ≈ s + 1
        @test @inferred(shift!(M, [NaN], [one(s)])) ≈ [s + 1]

        @test_opt getshift(M)
        @test_opt unshift(M, [s])
        @test_opt unshift(M, s)
        @test_opt unshift!(M, [NaN], [s])
        @test_opt shift(M, [s])
        @test_opt shift(M, s)
        @test_opt shift!(M, [NaN], [s])

        @test check_point(M, [s - 1]) === nothing
        @test check_point(M, [s]) isa DomainError
        @test check_point(M, [s + 10]) isa DomainError

        @test_opt check_point(M, [s - 1])
        @test_opt check_point(M, [s + 1])

        @test check_vector(M, [s - 1], [0]) === nothing
        @test check_vector(M, [s - 1], [s]) === nothing
        @test check_vector(M, [s - 1], [s - 1]) === nothing
        @test check_vector(M, [s - 1], [s + 1]) === nothing
        @test check_vector(M, [s - 1], [100]) === nothing
        @test check_vector(M, [s - 1], [-100]) === nothing

        @test_opt check_vector(M, [s - 1], [0])
        @test_opt check_vector(M, [s - 1], [s])

        # Test that those functions are not allocating
        @test @eval(@allocated(representation_size($M))) === 0
        @test @eval(@allocated(manifold_dimension($M))) === 0
        @test @eval(@allocated(injectivity_radius($M))) === 0
        @test @eval(@allocated(is_flat($M))) === 0
        @test @eval(@allocated(getshift($M))) === 0

        p = [s - 1]
        X = [1]
        Y = [1]

        @test_opt inner(M, p, X, Y)
        @test @eval(@allocated(inner($M, $p, $X, $Y))) === 0

        p = [s]
        c = [NaN]
        @test @allocated(shift!(M, c, p)) === 0
        @test @allocated(unshift!(M, c, p)) === 0
    end
end

@testitem "Double check that certain functionality is not allocating when used with StaticArrays" begin
    import ExponentialFamilyManifolds:
        ShiftedNegativeNumbers, getshift, shift, unshift, shift!, unshift!
    using ManifoldsBase, Static, StaticArrays, Manifolds

    # `Test` should pre-compile the call in the local functions 
    # in order to be non-allocating (probs could also use `@eval`)
    function foo()
        M = ShiftedNegativeNumbers(0)
        p = @SArray([0.0])
        X = @SArray([1.0])
        Y = @SArray([1.0])

        # `@test` macro allocates
        @assert shift(M, p) == p
        @assert unshift(M, p) == p

        s = unshift(M, shift(M, p))

        @assert iszero(s[1])

        i = inner(M, p, X, Y)

        @assert i ≈ inner(PositiveNumbers(), -p[1], X[1], Y[1])

        return s[1], i
    end

    # precompile
    foo()

    @test @allocated(foo()) === 0
end

@testitem "Manifolds.test_manifold" begin
    using Manifolds, Static, Random, StaticArrays

    import ExponentialFamilyManifolds: ShiftedNegativeNumbers

    shifts = [0, 0.0, 0.0f0, 1, 1.0, 1.0f0, static(-1), static(1), rand(), randn()]

    for s in shifts
        M = ShiftedNegativeNumbers(s)

        ptss = [
            [[s - 1], [s - 2], [s - 3]],
            [@SVector([s - 1]), @SVector([s - 2]), @SVector([s - 3])],
            [rand(M) for _ in 1:10],
        ]

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
            )
        end
    end
end

@testitem "Simple manifold optimization problem #1" begin
    using Manopt, ForwardDiff, Static, StableRNGs, LinearAlgebra

    import ExponentialFamilyManifolds: ShiftedNegativeNumbers

    for a in (2.0, 3.0),
        b in (10.0, 5.0),
        c in (1.0, 10.0, -1.0),
        eps in (1e-4, 1e-5, 1e-8, 1e-10),
        stepsize in (ConstantLength(0.1), ConstantLength(0.01), ConstantLength(0.001))

        expected_q = -b / 2a
        expected_minimum = c - b^2 / (4a)

        f(M, x) = (a .* x .^ 2 .+ b .* x .+ c)[1]

        grad_f(M, x) = 2 .* a .* x .+ b

        rng = StableRNG(42)

        for s in [0, 0.0, expected_q + 10, static(0), static(expected_q + 1)]
            M = ShiftedNegativeNumbers(s)
            p0 = rand(rng, M)

            q1 = gradient_descent(
                M,
                f,
                grad_f,
                p0;
                stepsize=stepsize,
                stopping_criterion=StopWhenGradientNormLess(eps) |
                                   StopAfterIteration(1_000_000),
            )

            @test q1[1] ≈ expected_q rtol = 2eps
            @test f(M, q1)[1] ≈ expected_minimum rtol = 2eps
            @test norm(M, f(M, q1), grad_f(M, q1)) <= 10eps # adjusted to stepsize
        end
    end
end

@testitem "Simple manifold optimization problem with inplace evaluation #2" begin
    using Manopt, ForwardDiff, Static, StableRNGs, LinearAlgebra, Manifolds, JET, Test

    import ExponentialFamilyManifolds: ShiftedNegativeNumbers

    include("manopt_setuptests.jl")

    a = 2.0
    b = 10.0
    c = 10.0

    expected_q = -b / 2a
    expected_minimum = c - b^2 / (4a)

    function f(M, x)
        a = 2.0
        b = 10.0
        c = 10.0
        return a .* x .^ 2 .+ b .* x .+ c
    end

    function grad_f!(M, X, p)
        a = 2.0
        b = 10.0
        c = 10.0
        return @. X = 2 * a * p + b
    end

    rng = StableRNG(42)
    M = ShiftedNegativeNumbers(static(0))
    p0 = rand(rng, M)
    X = zero_vector(M, p0)

    @test_opt f(M, p0)
    @test_opt grad_f!(M, X, p0)

    function prepare_state(M, p0)
        q = copy(p0)
        obj = ManifoldGradientObjective(missing, grad_f!; evaluation=InplaceEvaluation())
        dmp = DefaultManoptProblem(M, obj)
        s = GradientDescentState(
            M;
            p = q,
            stopping_criterion=StopWhenGradientNormLessNonAllocating(1e-8),
            stepsize=ConstantStepsizeNonAllocating(0.1),
            direction=Manopt.IdentityUpdateRule(),
            retraction_method=default_retraction_method(M, typeof(q)),
            X=zero_vector(M, q),
        )
        return dmp, s
    end

    function compute!(dmp, s)
        solve!(dmp, s)
        return Manopt.get_solver_return(Manopt.get_objective(dmp), s)
    end

    dmp, s = prepare_state(M, p0)
    q1 = compute!(dmp, s)

    dmp, s = prepare_state(M, p0)
    @test_opt compute!(dmp, s)

    dmp, s = prepare_state(M, p0)
    @test @allocated(compute!(dmp, s)) === 0

    function grad_f(M, p)
        return grad_f!(M, similar(p), p)
    end

    @test q1[1] ≈ expected_q rtol = 1e-8
    @test f(M, q1)[1] ≈ expected_minimum rtol = 1e-6
    @test norm(M, f(M, q1), grad_f(M, q1)) <= 1e-8
end
