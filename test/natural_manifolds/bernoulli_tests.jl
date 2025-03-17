@testitem "Check `Bernoulli` natural manifold" begin
    include("natural_manifolds_setuptests.jl")

    test_natural_manifold() do rng
        return Bernoulli(rand(rng))
    end
end

@testitem "Check SecondOrderRetraction" begin
    include("natural_manifolds_setuptests.jl")

    using ADTypes: AutoForwardDiff

    rng = StableRNG(42)
    M = ExponentialFamilyManifolds.get_natural_manifold(Bernoulli, ())
    p = rand(rng, M)
    X = rand(rng, M)
    basis = ExponentialFamilyManifolds.NaturalBasis()

    @show Manifolds.local_metric(M, p, basis)
    @show Manifolds.local_metric_jacobian(M, p, basis, backend=AutoForwardDiff())
    q = retract(
        M,
        p,
        X,
        ExponentialFamilyManifolds.SecondOrderRetraction(; backend=AutoForwardDiff()),
    )
    @show q
end
