using ManifoldsBase
using Random

"""
    SinglePointManifold(point)

This manifold represents a set from one point.
"""
struct SinglePointManifold{T, R} <: AbstractManifold{ℝ}
    point::T
    representation_size::R
end

function SinglePointManifold(point::T) where {T}
    return SinglePointManifold(point, size(point))
end

function Base.show(io::IO, M::SinglePointManifold)
    print(io, "SinglePointManifold(", M.point, ")")
end

ManifoldsBase.manifold_dimension(::SinglePointManifold) = 0
ManifoldsBase.representation_size(M::SinglePointManifold) = M.representation_size
ManifoldsBase.injectivity_radius(M::SinglePointManifold) = zero(eltype(M.point))

ManifoldsBase.default_retraction_method(::SinglePointManifold) = ExponentialRetraction()

function ManifoldsBase.check_point(M::SinglePointManifold, p; kwargs...)
    if !(p ≈ M.point)
        return DomainError(p, "The point $(p) does not lie on $(M), which contains only $(M.point).")
    end
    return nothing
end

function ManifoldsBase.check_vector(M::SinglePointManifold, p, X; kwargs...)
    if !iszero(X) && size(M.point) == size(X)
        return DomainError(X, "The tangent space of $(M) contains only the zero vector.")
    end
    return nothing
end

ManifoldsBase.is_flat(::SinglePointManifold) = true

ManifoldsBase.embed(::SinglePointManifold, p) = p
ManifoldsBase.embed(::SinglePointManifold, p, X) = X

function ManifoldsBase.inner(::SinglePointManifold, p, X, Y)
    return zero(eltype(X))
end

function ManifoldsBase.exp!(M::SinglePointManifold, q, p, X, t::Number=1)
    q .= M.point
    return q
end

function ManifoldsBase.log!(::SinglePointManifold, X, p, q)
    X .= zero(eltype(X))
    return X
end

function ManifoldsBase.project!(::SinglePointManifold, Y, p, X)
    fill!(Y, zero(eltype(Y)))
    return Y
end

function ManifoldsBase.zero_vector!(::SinglePointManifold, X, p)
    return fill!(X, zero(eltype(X)))
end

function Random.rand(M::SinglePointManifold; kwargs...)
    return rand(Random.default_rng(), M; kwargs...)
end

function Random.rand(rng::AbstractRNG, M::SinglePointManifold; kwargs...)
    return M.point
end
