"""
    ShiftedPositiveNumbers(shift)

A manifold representing the positive numbers shifted by `shift`. 
The points on this manifold are 1-dimensional vectors with a single element.
"""
struct ShiftedPositiveNumbers{S} <: AbstractManifold{ℝ}
    shift::S
end

function Base.show(io::IO, M::ShiftedPositiveNumbers)
    return print(io, "ShiftedPositiveNumbers(", getshift(M), ")")
end

getshift(M::ShiftedPositiveNumbers) = M.shift

unshift(M::ShiftedPositiveNumbers, p) = p .- M.shift
unshift(M::ShiftedPositiveNumbers, p::Real) = p - M.shift
unshift!(M::ShiftedPositiveNumbers, X, p) = map!(Base.Fix2(-, M.shift), X, p)

shift(M::ShiftedPositiveNumbers, p) = p .+ M.shift
shift(M::ShiftedPositiveNumbers, p::Real) = p + M.shift
shift!(M::ShiftedPositiveNumbers, X, p) = map!(Base.Fix2(+, M.shift), X, p)

ManifoldsBase.get_embedding(::ShiftedPositiveNumbers) = Euclidean(1; field=ℝ)

ManifoldsBase.representation_size(::ShiftedPositiveNumbers) = (1,)
ManifoldsBase.manifold_dimension(::ShiftedPositiveNumbers) = 1
function ManifoldsBase.injectivity_radius(::ShiftedPositiveNumbers)
    return ManifoldsBase.injectivity_radius(PositiveNumbers())
end
ManifoldsBase.is_flat(::ShiftedPositiveNumbers) = ManifoldsBase.is_flat(PositiveNumbers())

function ManifoldsBase.check_point(M::ShiftedPositiveNumbers, p; kwargs...)
    if p[1] <= getshift(M)
        return DomainError(
            p,
            lazy"The point $(p) does not lie on $(M), since it is less or equal to than $(getshift(M)).",
        )
    end
    return nothing
end

function ManifoldsBase.check_vector(M::ShiftedPositiveNumbers, p, X; kwargs...)
    # The whole real line is fine as a tangent
    return nothing
end

ManifoldsBase.embed(::ShiftedPositiveNumbers, p) = p
ManifoldsBase.embed(::ShiftedPositiveNumbers, p, X) = X

function ManifoldsBase.inner(M::ShiftedPositiveNumbers, p, X, Y)
    return ManifoldsBase.inner(
        PositiveNumbers(), unshift(M, @inbounds(p[1])), @inbounds(X[1]), @inbounds(Y[1])
    )
end

function ManifoldsBase.exp_fused!(M::ShiftedPositiveNumbers, q, p, X, t::Number)
    @inbounds q[1] = shift(
        M,
        ManifoldsBase.exp(
            PositiveNumbers(), 
            unshift(M, @inbounds(p[1])), 
            @inbounds(X[1]) * t  # Scale the tangent vector by t
        ),
    )
    return q
end

function ManifoldsBase.exp!(M::ShiftedPositiveNumbers, q, p, X)
    return ManifoldsBase.exp_fused!(M, q, p, X, one(eltype(p)))
end

function ManifoldsBase.log!(M::ShiftedPositiveNumbers, X, p, q)
    @inbounds X[1] = ManifoldsBase.log(
        PositiveNumbers(), unshift(M, @inbounds(p[1])), unshift(M, @inbounds(q[1]))
    )
    return X
end

function ManifoldsBase.project!(M::ShiftedPositiveNumbers, Y, p, X)
    @inbounds Y[1] = project(
        PositiveNumbers(), unshift(M, @inbounds(p[1])), @inbounds(X[1])
    )
    return Y
end

ManifoldsBase.default_retraction_method(::ShiftedPositiveNumbers) = ExponentialRetraction()

function ManifoldsBase.retract_fused!(
    M::ShiftedPositiveNumbers, q, p, X, t::Number, ::ExponentialRetraction
)
    return ManifoldsBase.exp_fused!(M, q, p, X, t)
end

function ManifoldsBase.retract!(M::ShiftedPositiveNumbers, q, p, X, ::ExponentialRetraction)
    return ManifoldsBase.retract_fused!(M, q, p, X, one(eltype(p)), ExponentialRetraction())
end

function ManifoldsBase.parallel_transport_to!(M::ShiftedPositiveNumbers, Y, p, X, q)
    return parallel_transport_to!(
        PositiveNumbers(), Y, unshift(M, @inbounds(p[1])), X, unshift(M, @inbounds(q[1]))
    )
end

ManifoldsBase.zero_vector!(::ShiftedPositiveNumbers, X, p) = fill!(X, 0)

function Random.rand(M::ShiftedPositiveNumbers; kwargs...)
    return rand(Random.default_rng(), M; kwargs...)
end

function Random.rand(rng::AbstractRNG, M::ShiftedPositiveNumbers; kwargs...)
    _p = shift(M, rand(rng, PositiveNumbers(); kwargs...))
    return [_p]
end

function Random.rand!(rng::AbstractRNG, M::ShiftedPositiveNumbers, pX; kwargs...)
    Random.rand!(rng, PositiveNumbers(), pX; kwargs...)
    shift!(M, pX, pX)
    return nothing
end