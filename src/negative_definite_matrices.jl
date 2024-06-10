
struct NegativeDefiniteMatrices{B} <: AbstractManifold{ℝ}
    base::B

    function NegativeDefiniteMatrices(size)
        base = SymmetricPositiveDefinite(size)
        return new{typeof(base)}(base)
    end
end

function Base.show(io::IO, M::NegativeDefiniteMatrices)
    print(io, "NegativeDefiniteMatrices(", first(representation_size(M.base)), ")")
end

ManifoldsBase.get_embedding(M::NegativeDefiniteMatrices) =
    ManifoldsBase.get_embedding(M.base)

ManifoldsBase.representation_size(M::NegativeDefiniteMatrices) =
    ManifoldsBase.representation_size(M.base)
ManifoldsBase.manifold_dimension(M::NegativeDefiniteMatrices) =
    ManifoldsBase.manifold_dimension(M.base)
ManifoldsBase.injectivity_radius(M::NegativeDefiniteMatrices) =
    ManifoldsBase.injectivity_radius(M.base)
ManifoldsBase.is_flat(M::NegativeDefiniteMatrices) = ManifoldsBase.is_flat(M.base)

function ManifoldsBase.check_point(M::NegativeDefiniteMatrices, p; kwargs...)
    if !isapprox(norm(p - transpose(p)), 0.0; kwargs...)
        return DomainError(
            norm(p - transpose(p)),
            lazy"The point $(p) does not lie on $(M) since its not a symmetric matrix:",
        )
    end
    if !isposdef(Negated(p))
        return DomainError(
            eigvals(p),
            lazy"The point $p does not lie on $(M) since its not a negative definite matrix.",
        )
    end
    return nothing
end

function ManifoldsBase.check_vector(M::NegativeDefiniteMatrices, p, X; kwargs...)
    if !isapprox(X, transpose(X); kwargs...)
        return DomainError(
            X,
            lazy"The vector $(X) is not a tangent to a point on $(M) (represented as an element of the Lie algebra) since its not symmetric.",
        )
    end
    return nothing
end

ManifoldsBase.embed(::NegativeDefiniteMatrices, p) = p
ManifoldsBase.embed(::NegativeDefiniteMatrices, p, X) = X

function ManifoldsBase.inner(M::NegativeDefiniteMatrices, p, X, Y)
    return ManifoldsBase.inner(M.base, Negated(p), Negated(X), Negated(Y))
end

function ManifoldsBase.exp!(M::NegativeDefiniteMatrices, q, p, X, t::Number = 1)
    ManifoldsBase.exp!(M.base, q, Negated(p), Negated(X), t)
    return negate!(q)
end

function ManifoldsBase.log!(M::NegativeDefiniteMatrices, X, p, q)
    ManifoldsBase.log!(M.base, X, Negated(p), Negated(q))
    return negate!(X)
end

function ManifoldsBase.project!(M::NegativeDefiniteMatrices, Y, p, X)
    ManifoldsBase.project!(M.base, Y, Negated(p), Negated(X))
    return negate!(Y)
end

ManifoldsBase.default_retraction_method(::NegativeDefiniteMatrices) =
    ExponentialRetraction()

ManifoldsBase.retract!(
    M::NegativeDefiniteMatrices,
    q,
    p,
    X,
    t::Number,
    ::ExponentialRetraction,
) = ManifoldsBase.exp!(M, q, p, X, t)

function ManifoldsBase.parallel_transport_to!(M::NegativeDefiniteMatrices, Y, p, X, q)
    parallel_transport_to!(M.base, Y, Negated(p), Negated(X), Negated(q))
    return negate!(Y)
end

ManifoldsBase.zero_vector!(::NegativeDefiniteMatrices, X, p) = fill!(X, 0)

function Random.rand(M::NegativeDefiniteMatrices; vector_at = nothing, kwargs...)
    return rand(Random.default_rng(), M; vector_at = vector_at, kwargs...)
end

function Random.rand(
    rng::AbstractRNG,
    M::NegativeDefiniteMatrices;
    vector_at = nothing,
    kwargs...,
)
    return negate!(
        rand(
            rng,
            M.base;
            vector_at = isnothing(vector_at) ? nothing : Negated(vector_at),
            kwargs...,
        ),
    )
end

function Random.rand!(
    rng::AbstractRNG,
    M::NegativeDefiniteMatrices,
    pX;
    vector_at = nothing,
    kwargs...,
)
    return negate!(
        Random.rand!(
            rng,
            M.base,
            pX;
            vector_at = isnothing(vector_at) ? nothing : Negated(vector_at),
            kwargs...,
        ),
    )
end

# Utility functions

negate!(p) = map!(Base.Fix2(*, -1), p, p)

"""
    Negated(m)

Lazily negates the matrix `m`, without creating a new matrix. 
Works by redefining the `getindex`.
"""
struct Negated{T,M} <: AbstractMatrix{T}
    m::M
end

function Negated(m::M) where {T,M<:AbstractMatrix{T}}
    return Negated{T,M}(m)
end

Base.show(io::IO, N::Negated) = print(io, "Negated(", N.m, ")")

Base.IteratorEltype(::Type{<:Negated{T,M}}) where {T,M} = Base.IteratorEltype(M)
Base.IteratorSize(::Type{<:Negated{T,M}}) where {T,M} = Base.IteratorSize(M)

function Base.iterate(N::Negated)
    iter = Base.iterate(N.m)
    if isnothing(iter)
        return nothing
    end
    item, state = iter
    return (-item, state)
end

function Base.iterate(N::Negated, state)
    iter = Base.iterate(N.m, state)
    if isnothing(iter)
        return nothing
    end
    item, state = iter
    return (-item, state)
end

Base.copy(N::Negated) = Negated(Base.copy(N.m))
Base.unaliascopy(N::Negated) = Negated(Base.unaliascopy(N.m))
Base.size(N::Negated) = Base.size(N.m)
Base.length(N::Negated) = Base.length(N.m)
Base.eltype(::Type{<:Negated{T}}) where {T} = T
Base.isdone(N::Negated) = Base.isdone(N.m)
Base.isdone(N::Negated, state) = Base.isdone(N.m, state)

Base.@propagate_inbounds Base.getindex(N::Negated, index...) = -getindex(N.m, index...)