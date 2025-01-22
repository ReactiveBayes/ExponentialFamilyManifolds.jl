@testitem "Partial derivatives of NormalGamma Fisher metric in natural coords" begin

    import ExponentialFamilyManifolds: partial_metric
    using ExponentialFamily
    using ForwardDiff
    using BenchmarkTools

    # define some test points in domain:
    test_points = [
        [1.0, -2.0, 0.5, -1.0],
        [1.5, -1.0, 0.3, -0.9],
        [2.0, -3.0, 1.0, -2.0],
        [0.5, -0.5, 0.1, -0.1],
    ]

    d = NormalGamma(1.0, 1.0, 1.0, 1.0)
    ef = convert(ExponentialFamilyDistribution, d)
    locfisher = (η) -> fisherinformation(ef, η)


     # A helper that flattens
    function fisher_metric_vec(η)
        G = locfisher(η)
        return vec(G)
    end

    function forwarddiff_partials(η)
        FD_jac = ForwardDiff.jacobian(fisher_metric_vec, η)
        out = Array{Float64}(undef, 4,4,4)
        for i in 1:4
            for j in 1:4
                r = 4*(i-1)+j
                for c in 1:4
                    out[i,j,c] = FD_jac[r,c]
                end
            end
        end
        return out
    end

    for η in test_points
        fd_partials = forwarddiff_partials(η)
        sym_partials = partial_metric(NormalGamma, NaturalParametersSpace(), η)

        diff_array = fd_partials .- sym_partials
        err = maximum(abs, diff_array)

        @test err < 1e-6  # or smaller threshold if you want more precision

        fd_bench = @benchmark forwarddiff_partials($η)
        sym_bench = @benchmark partial_metric(NormalGamma, NaturalParametersSpace(), $η)
        @test min(sym_bench.times...) < min(fd_bench.times...)
        #TODO WHY?
        # probably smt just wrong here
        @test_broken sym_bench.allocs < fd_bench.allocs
    end
end

@testitem "Generic properties of NormalGammaNaturalManifold" begin
    import ExponentialFamilyManifolds: NormalGammaNaturalManifold
    import ManifoldsBase: manifold_dimension, representation_size, check_point, check_vector, is_point, is_vector
    import ManifoldsBase: zero_vector, inner
    import ExponentialFamilyManifolds: isproper, NaturalParametersSpace

    using ManifoldsBase, Test
    using LinearAlgebra
    using ExponentialFamily

    M = NormalGammaNaturalManifold()
    valid_point = [1.0, -2.0, 0.5, -1.0]
    ef = ExponentialFamilyDistribution(NormalGamma, [1.0, -2.0, 0.5, -1.0], nothing)

    @test @inferred(manifold_dimension(M)) === 4
    @test @inferred(representation_size(M)) === (4,)
    @test injectivity_radius(M) === Inf

    @test check_point(M, valid_point) === nothing
    @test check_point(M, [2.0, 0.0, 1.0, -3.0]) isa DomainError # invalid because η2>=0
    @test check_point(M, [0.5, -3.0, -1.0, -1.0]) isa DomainError # invalid because η3 <= -1/2
    @test check_point(M, [1, 1, 1]) isa DomainError # invalid because because length(η) != 4
    @test check_point(M, [1, 1, 1, 1, 1]) isa DomainError # invalid because because length(η) != 4
     
    # Now define some tangent vectors and test them:
    tangent_vectors = [
        [0.0, 0.0, 0.0, 0.0],
        [0.1, -0.2, 0.3, -0.4],
        [1.0, 1.0, 1.0, 1.0]
    ]

    for X in tangent_vectors
        @test check_vector(M, valid_point, X) === nothing
    end

    @test check_vector(M, valid_point, [1.0, 2.0, 3.0]) isa DomainError
    @test check_vector(M, valid_point, [1.0, 2.0, 3.0, 4.0, 5.0]) isa DomainError

    z = zero_vector(M, valid_point)
    @test length(z) == 4
    @test all(x->x==0.0, z)
    
    X = [0.1, 0.0, -0.1, 0.05]
    Y = [0.2, -0.3, 0.0, -0.05]
    @test inner(M, valid_point, X, Y) ≈ dot(X, fisherinformation(ef), Y)
    
    @test @eval(@allocated(manifold_dimension($M))) === 0
    @test @eval(@allocated(representation_size($M))) === 0
end

@testitem "Simple manifold optimization problem #1" begin
    using Manopt, ForwardDiff, Static, StableRNGs, LinearAlgebra, Test
    using ExponentialFamily, ExponentialFamilyManifolds
    using BayesBase
    import ExponentialFamilyManifolds: NormalGammaNaturalManifold
    import ManifoldsBase: inner

    rng = StableRNG(42)
    p_true = (2.0, 1.5, 1.0, 3.0)

    dist = NormalGamma(p_true...)
    ef = convert(ExponentialFamilyDistribution, dist)
    η = getnaturalparameters(ef)
    data = rand(rng, dist, 200)

    M = NormalGammaNaturalManifold()

    function f(M, p)
        ef = ExponentialFamilyDistribution(NormalGamma, p, nothing)
        return -mean((x) -> logpdf(ef, x), data)
    end

    function grad_f(M, p)
        ef = ExponentialFamilyDistribution(NormalGamma, p, nothing)
        egrad = ForwardDiff.gradient((p) -> f(M, p), p)
        G = fisherinformation(ef)
        r = G \ egrad
        return r
    end

    p0 = [1.0, -2.0, 0.5, -1.0]

    p_opt = gradient_descent(
        M,
        f, 
        grad_f,
        p0;
        stepsize = ConstantLength(0.01),
        stopping_criterion = StopWhenGradientNormLess(1e-6),
        max_iterations = 500
    )

    cost_before = f(M, p0)
    cost_after  = f(M, p_opt)
    @test cost_after < cost_before
    final_grad = grad_f(M, p_opt)
    @test norm(final_grad) < 1e-5
end