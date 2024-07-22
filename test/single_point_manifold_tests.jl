@testitem "Generic properties of SinglePointManifold" begin
    import ManifoldsBase: check_point, check_vector, embed, representation_size, injectivity_radius, get_embedding, is_flat, inner, manifold_dimension
    import ExponentialFamilyManifolds: SinglePointManifold
    using ManifoldsBase, Static, StaticArrays, JET, Manifolds
    using StableRNGs

    rng = StableRNG(42)


    points = [
        0,
        0.0,
        0.0f0,
        1,
        1.0,
        1.0f0, 
        -1,
        2,
        π,
        rand(),
        randn()
    ]

    for p in points
        M = SinglePointManifold(p)

        @test repr(M) == "SinglePointManifold($p)"

        @test @inferred(representation_size(M)) === ()
        @test @inferred(manifold_dimension(M)) === 0
        @test @inferred(is_flat(M)) === true
        @test injectivity_radius(M) ≈ 0
     
        @test_throws MethodError get_embedding(M)

        @test check_point(M, p) === nothing
        @test check_point(M, p + 1) isa DomainError
        @test check_point(M, p - 1) isa DomainError

        @test check_vector(M, p, 0) === nothing
        @test check_vector(M, p, 1) isa DomainError
        @test check_vector(M, p, -1) isa DomainError

        @test @eval(@allocated(representation_size($M))) === 0
        @test @eval(@allocated(manifold_dimension($M))) === 0
        @test @eval(@allocated(is_flat($M))) === 0

        X = [1]
        Y = [1]

        @test_opt inner(M, p, X, Y)
        @test_opt inner(M, p, 0, 0)

        @test embed(M, p) == p
        @test embed(M, p, 0) == 0
        @test inner(M, p, 0, 0) == 0
    end

    vector_points = [[1], [1, 2], [1, 2, 3]]

    for p in vector_points
        M = SinglePointManifold(p)
        q = similar(p)
        X = zero_vector(M, p)
        @test ManifoldsBase.exp!(M, q, p, X) == p
        @test ManifoldsBase.log!(M, X, p, p) == zero_vector(M, p)
        @test ManifoldsBase.log(M, p, p) == zero_vector(M, p)
        @test ManifoldsBase.project!(M, similar(X), p, similar(X)) == zero_vector(M, p)
        @test rand(rng, M) ∈ M
    end
end
