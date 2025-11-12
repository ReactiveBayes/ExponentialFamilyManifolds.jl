
"""
    get_natural_manifold_base(::Type{NormalMeanVariance}, ::Tuple{}, conditioner = nothing)

Get the natural manifold base for the `NormalMeanVariance` distribution.
"""
function get_natural_manifold_base(
    ::Type{NormalMeanVariance}, ::Tuple{}, conditioner=nothing
)
    return ProductManifold(Euclidean(1), PositiveVectors(1))
end

"""
    partition_point(::Type{NormalMeanVariance}, ::Tuple{}, p, conditioner = nothing)

Converts the `point` to a compatible representation for the natural manifold of type `NormalMeanVariance`.
"""
function partition_point(::Type{NormalMeanVariance}, ::Tuple{}, p, conditioner=nothing)
    return ArrayPartition(view(p, 1:1), -view(p, 2:2))
end

"""
    transform_back!(p, M::NaturalParametersManifold{ℝ, NormalMeanVariance}, q)

Transforms the `q` to a compatible representation for the exponential family distribution of type `NormalMeanVariance`.
"""
function transform_back!(p, ::NaturalParametersManifold{ℝ,NormalMeanVariance}, q)
    p[1:1] .= 1.0 .* view(q, 1:1)
    p[2:2] .= -view(q, 2:2)
    return p
end

"""
    get_natural_manifold_base(::Type{MvNormalMeanCovariance}, dims::Tuple{Int}, conditioner = nothing)

Get the natural manifold base for the `MvNormalMeanCovariance` distribution.
"""
function get_natural_manifold_base(
    ::Type{MvNormalMeanCovariance}, dims::Tuple{Int}, conditioner=nothing
)
    k = first(dims)
    return ProductManifold(Euclidean(k), SymmetricPositiveDefinite(k))
end

# Guard for missing dimension in MvNormalMeanCovariance
"""
    get_natural_manifold_base(::Type{MvNormalMeanCovariance}, ::Tuple{}, conditioner=nothing)

Raises an ArgumentError when the manifold is requested without the required
dimension tuple `(K,)`. The multivariate normal distribution needs the number
of variables `K`.
"""
function get_natural_manifold_base(
    ::Type{MvNormalMeanCovariance}, ::Tuple{}, conditioner=nothing
)
    throw(
        ArgumentError(
            "MvNormalMeanCovariance requires an explicit dimension `(K,)`. " *
            "Example: `get_natural_manifold(MvNormalMeanCovariance, (3,))`.",
        ),
    )
end

"""
    partition_point(::Type{MvNormalMeanCovariance}, dims::Tuple{Int}, p, conditioner = nothing)

Converts the `point` to a compatible representation for the natural manifold of type `MvNormalMeanCovariance`.
"""
function partition_point(
    ::Type{MvNormalMeanCovariance}, dims::Tuple{Int}, p, conditioner=nothing
)
    k = first(dims)
    return ArrayPartition(view(p, 1:k), -reshape(view(p, (k + 1):(k + k ^ 2)), (k, k)))
end

"""
    partition_point(::Type{MvNormalMeanCovariance}, ::Tuple{}, p, conditioner=nothing)

Guard that throws an ArgumentError for missing dimension.
"""
function partition_point(::Type{MvNormalMeanCovariance}, ::Tuple{}, p, conditioner=nothing)
    throw(
        ArgumentError(
            "MvNormalMeanCovariance requires a dimension `(K,)` for partitioning points. " *
            "Example: `get_natural_manifold(MvNormalMeanCovariance, (3,))`.",
        ),
    )
end

"""
    transform_back!(p, M::NaturalParametersManifold{ℝ, MvNormalMeanCovariance}, q)

Transforms the `q` to a compatible representation for the exponential family distribution of type `MvNormalMeanCovariance`.
"""
function transform_back!(p, M::NaturalParametersManifold{ℝ,MvNormalMeanCovariance}, q)
    k = first(getdims(M))
    p[1:k] .= 1.0 .* view(q, 1:k)
    p[(k + 1):(k + k ^ 2)] .= -view(q, (k + 1):(k + k ^ 2))
    return p
end

"""
    get_natural_manifold_base(::Type{MvNormalMeanScalePrecision}, dims::Tuple{Int}, conditioner = nothing)

Get the natural manifold base for the `MvNormalMeanScalePrecision` distribution.
"""
function get_natural_manifold_base(
    ::Type{MvNormalMeanScalePrecision}, dims::Tuple{Int}, conditioner=nothing
)
    k = first(dims)
    return ProductManifold(Euclidean(k), PositiveVectors(1))
end

# Guard for missing dimension in MvNormalMeanScalePrecision
"""
    get_natural_manifold_base(::Type{MvNormalMeanScalePrecision}, ::Tuple{}, conditioner=nothing)

Raises an ArgumentError when called without dimension tuple `(K,)`.
"""
function get_natural_manifold_base(
    ::Type{MvNormalMeanScalePrecision}, ::Tuple{}, conditioner=nothing
)
    throw(
        ArgumentError(
            "MvNormalMeanScalePrecision requires an explicit dimension `(K,)`. " *
            "Example: `get_natural_manifold(MvNormalMeanScalePrecision, (3,))`.",
        ),
    )
end

"""
    partition_point(::Type{MvNormalMeanCovariance}, dims::Tuple{Int}, p, conditioner = nothing)

Converts the `point` to a compatible representation for the natural manifold of type `MvNormalMeanCovariance`.
"""
function partition_point(
    ::Type{MvNormalMeanScalePrecision}, dims::Tuple{Int}, p, conditioner=nothing
)
    k = first(dims)
    return ArrayPartition(view(p, 1:k), -view(p, (k + 1):(k + 1)))
end

"""
    partition_point(::Type{MvNormalMeanScalePrecision}, ::Tuple{}, p, conditioner=nothing)

Guard that throws an ArgumentError for missing dimension.
"""
function partition_point(
    ::Type{MvNormalMeanScalePrecision}, ::Tuple{}, p, conditioner=nothing
)
    throw(
        ArgumentError(
            "MvNormalMeanScalePrecision requires a dimension `(K,)` for partitioning points. " *
            "Example: `get_natural_manifold(MvNormalMeanScalePrecision, (3,))`.",
        ),
    )
end

"""
    transform_back!(p, M::NaturalParametersManifold{ℝ, MvNormalMeanScalePrecision}, q)

Transforms the `q` to a compatible representation for the exponential family distribution of type `MvNormalMeanScalePrecision`.
"""
function transform_back!(p, M::NaturalParametersManifold{ℝ,MvNormalMeanScalePrecision}, q)
    k = first(getdims(M))
    p[1:k] .= 1.0 .* view(q, 1:k)
    p[(k + 1):(k + 1)] .= -view(q, (k + 1):(k + 1))
    return p
end
