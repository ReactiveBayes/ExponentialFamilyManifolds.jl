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
        return (mean(p) - 0.5)^2
    end

    function g(M, p)
        X = ForwardDiff.gradient((p) -> f(M, p), p)
        return project!(M, X, p, X)
    end

    q = gradient_descent(M, f, g, rand(rng, M))
    @show is_point(M, q)
    @test mean(q) ≈ 0.5 atol = 1e-1
end