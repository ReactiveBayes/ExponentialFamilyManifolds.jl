@testitem "Check `Wishart` natural manifold" begin
    include("natural_manifolds_setuptests.jl")
    import ExponentialFamily: WishartFast
    test_natural_manifold() do rng
        k = rand(rng, 2:10)
        L = LowerTriangular(randn(rng, k, k))
        C = L * L' + k * I
        return WishartFast(k + 2, C)
    end
end
