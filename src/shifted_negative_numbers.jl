"""
    ShiftedNegativeNumbers(shift)

A manifold representing the negative numbers shifted by `shift`.
The points on this manifold are 1-dimensional vectors with a single element.
"""
struct ShiftedNegativeNumbers{S} <: AbstractManifold{ℝ}
    shift::S
end

function Base.show(io::IO, M::ShiftedNegativeNumbers)
    return print(io, "ShiftedNegativeNumbers(", getshift(M), ")")
end

getshift(M::ShiftedNegativeNumbers) = M.shift

unshift(M::ShiftedNegativeNumbers, p) = p .- M.shift
unshift(M::ShiftedNegativeNumbers, p::Real) = p - M.shift
unshift!(M::ShiftedNegativeNumbers, X, p) = map!(Base.Fix2(-, M.shift), X, p)

shift(M::ShiftedNegativeNumbers, p) = p .+ M.shift
shift(M::ShiftedNegativeNumbers, p::Real) = p + M.shift
shift!(M::ShiftedNegativeNumbers, X, p) = map!(Base.Fix2(+, M.shift), X, p)

ManifoldsBase.get_embedding(::ShiftedNegativeNumbers) = Euclidean(1; field=ℝ)

ManifoldsBase.representation_size(::ShiftedNegativeNumbers) = (1,)
ManifoldsBase.manifold_dimension(::ShiftedNegativeNumbers) = 1
function ManifoldsBase.injectivity_radius(::ShiftedNegativeNumbers)
    return ManifoldsBase.injectivity_radius(PositiveNumbers())
end
ManifoldsBase.is_flat(::ShiftedNegativeNumbers) = ManifoldsBase.is_flat(PositiveNumbers())

function ManifoldsBase.check_point(M::ShiftedNegativeNumbers, p; kwargs...)
    if p[1] >= getshift(M)
        return DomainError(
            p,
            lazy"The point $(p) does not lie on $(M), since it is less or equal to than $(getshift(M)).",
        )
    end
    return nothing
end

function ManifoldsBase.check_vector(M::ShiftedNegativeNumbers, p, X; kwargs...)
    # The whole real line is fine as a tangent
    return nothing
end

ManifoldsBase.embed(::ShiftedNegativeNumbers, p) = p
ManifoldsBase.embed(::ShiftedNegativeNumbers, p, X) = X

function ManifoldsBase.inner(M::ShiftedNegativeNumbers, p, X, Y)
    return ManifoldsBase.inner(
        PositiveNumbers(), -unshift(M, @inbounds(p[1])), -@inbounds(X[1]), -@inbounds(Y[1])
    )
end

function ManifoldsBase.exp_fused!(M::ShiftedNegativeNumbers, q, p, X, t::Number)
    @inbounds q[1] = shift(
        M,
        -ManifoldsBase.exp_fused(
            PositiveNumbers(), -unshift(M, @inbounds(p[1])), -@inbounds(X[1]), t
        ),
    )
    return q
end

function ManifoldsBase.exp!(M::ShiftedNegativeNumbers, q, p, X)
    return ManifoldsBase.exp_fused!(M, q, p, X, one(eltype(p)))
end

function ManifoldsBase.log!(M::ShiftedNegativeNumbers, X, p, q)
    @inbounds X[1] =
        -ManifoldsBase.log(
            PositiveNumbers(), -unshift(M, @inbounds(p[1])), -unshift(M, @inbounds(q[1]))
        )
    return X
end

function ManifoldsBase.project!(M::ShiftedNegativeNumbers, Y, p, X)
    @inbounds Y[1] =
        -project(PositiveNumbers(), -unshift(M, @inbounds(p[1])), -@inbounds(X[1]))
    return Y
end

ManifoldsBase.default_retraction_method(::ShiftedNegativeNumbers) = ExponentialRetraction()

function ManifoldsBase.retract_fused!(
    M::ShiftedNegativeNumbers, q, p, X, t::Number, ::ExponentialRetraction
)
    return ManifoldsBase.exp_fused!(M, q, p, X, t)
end

function ManifoldsBase.retract!(M::ShiftedNegativeNumbers, q, p, X, ::ExponentialRetraction)
    return ManifoldsBase.retract_fused!(M, q, p, X, one(eltype(p)), ExponentialRetraction())
end

function ManifoldsBase.parallel_transport_to!(M::ShiftedNegativeNumbers, Y, p, X, q)
    parallel_transport_to!(
        PositiveNumbers(), Y, -unshift(M, @inbounds(p[1])), -X, -unshift(M, @inbounds(q[1]))
    )
    map!(Base.Fix2(*, -1), Y, Y)
    return Y
end

ManifoldsBase.zero_vector!(::ShiftedNegativeNumbers, X, p) = fill!(X, 0)

function Random.rand(M::ShiftedNegativeNumbers; kwargs...)
    return rand(Random.default_rng(), M; kwargs...)
end

function Random.rand(rng::AbstractRNG, M::ShiftedNegativeNumbers; kwargs...)
    _p = shift(M, -rand(rng, PositiveNumbers(); kwargs...))
    return [_p]
end

function Random.rand!(rng::AbstractRNG, M::ShiftedNegativeNumbers, pX; kwargs...)
    Random.rand!(rng, PositiveNumbers(), pX; kwargs...)
    map!(Base.Fix2(*, -1), pX, pX)
    shift!(M, pX, pX)
    return nothing
end
