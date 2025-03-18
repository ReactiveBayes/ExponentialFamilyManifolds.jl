@testitem "Check `Categorical` natural manifold" begin
    include("natural_manifolds_setuptests.jl")

    test_natural_manifold() do rng
        p = rand(rng, 10)
        normalize!(p, 1)
        return Categorical(p)
    end
end

@testitem "Check that optimization work on Categorical" begin
    include("natural_manifolds_setuptests.jl")

    using Manopt, ForwardDiff
    using BayesBase

    rng = StableRNG(42)
    p = rand(StableRNG(42), 10)
    normalize!(p, 1)
    distribution = Categorical(p)
    sample = rand(rng, distribution)
    dims = size(sample)
    ef = convert(ExponentialFamilyDistribution, distribution)
    T = ExponentialFamily.exponential_family_typetag(ef)
    M = get_natural_manifold(T, dims, getconditioner(ef))

    function f(M, p)
        ef = convert(ExponentialFamilyDistribution, M, p)
        η = getnaturalparameters(ef)
        return (mean(η) - 0.5)^2
    end

    function g(M, p)
        return project(M, p, 2 * p ./ 10)
    end

    q = gradient_descent(M, f, g, rand(rng, M))
    @test q ∈ M
    @test mean(q) ≈ 0.5 atol = 1e-1
end

@testitem "Check that the manifold for `Categorical` interacts nicely with `ForwardDiff`" begin
    include("natural_manifolds_setuptests.jl")

    using ForwardDiff
    # JuliaDiff/ForwardDiff.jl#706
    rng = StableRNG(42)
    p = rand(rng, 10)
    normalize!(p, 1)
    distribution = Categorical(p)
    sample = rand(rng, distribution)
    dims = size(sample)
    ef = convert(ExponentialFamilyDistribution, distribution)
    T = ExponentialFamily.exponential_family_typetag(ef)
    M = get_natural_manifold(T, dims, getconditioner(ef))

    @test all(≈(1), ForwardDiff.gradient(sum, rand(rng, M)))
end

@testitem "Check MLE works for `Categorical`" begin
    include("natural_manifolds_setuptests.jl")

    using Manopt
    import Distributions: kldivergence, Distribution

    @test_broken test_mle_works(mle_samples=1000, ndistributions=3) do rng
        return Categorical(rand(rng, 3))
    end
end
