
"""
    get_natural_manifold_base(::Type{NormalMeanVariance}, ::Tuple{}, conditioner = nothing)

Get the natural manifold base for the `NormalMeanVariance` distribution.
"""
function get_natural_manifold_base(
    ::Type{NormalMeanVariance}, ::Tuple{}, conditioner=nothing
)
    return ProductManifold(Euclidean(1), ShiftedNegativeNumbers(static(0)))
end

"""
    partition_point(::Type{NormalMeanVariance}, ::Tuple{}, p, conditioner = nothing)

Converts the `point` to a compatible representation for the natural manifold of type `NormalMeanVariance`.
"""
function partition_point(::Type{NormalMeanVariance}, ::Tuple{}, p, conditioner=nothing)
    return ArrayPartition(view(p, 1:1), view(p, 2:2))
end

"""
    get_natural_manifold_base(::Type{MvNormalMeanCovariance}, dims::Tuple{Int}, conditioner = nothing)

Get the natural manifold base for the `MvNormalMeanCovariance` distribution.
"""
function get_natural_manifold_base(
    ::Type{MvNormalMeanCovariance}, dims::Tuple{Int}, conditioner=nothing
)
    k = first(dims)
    return ProductManifold(Euclidean(k), SymmetricNegativeDefinite(k))
end

"""
    partition_point(::Type{MvNormalMeanCovariance}, dims::Tuple{Int}, p, conditioner = nothing)

Converts the `point` to a compatible representation for the natural manifold of type `MvNormalMeanCovariance`.
"""
function partition_point(
    ::Type{MvNormalMeanCovariance}, dims::Tuple{Int}, p, conditioner=nothing
)
    k = first(dims)
    return ArrayPartition(view(p, 1:k), reshape(view(p, (k + 1):(k + k^2)), (k, k)))
end

"""
    get_natural_manifold_base(::Type{MvNormalMeanScalePrecision}, dims::Tuple{Int}, conditioner = nothing)

Get the natural manifold base for the `MvNormalMeanScalePrecision` distribution.
"""
function get_natural_manifold_base(
    ::Type{MvNormalMeanScalePrecision}, dims::Tuple{Int}, conditioner=nothing
)
    k = first(dims)
    return ProductManifold(Euclidean(k), ShiftedNegativeNumbers(static(0)))
end

"""
    partition_point(::Type{MvNormalMeanCovariance}, dims::Tuple{Int}, p, conditioner = nothing)

Converts the `point` to a compatible representation for the natural manifold of type `MvNormalMeanCovariance`.
"""
function partition_point(
    ::Type{MvNormalMeanScalePrecision}, dims::Tuple{Int}, p, conditioner=nothing
)
    k = first(dims)
    return ArrayPartition(view(p, 1:k), view(p, (k + 1):(k + 1)))
end

function getdefaultmetric(::Type{MvNormalMeanCovariance})
    return BaseMetric()
end

Manifolds.representation_size(::NaturalParametersManifold{𝔽, NormalMeanVariance}) where {𝔽} = (2,)

function Manifolds.representation_size(M::NaturalParametersManifold{𝔽, MvNormalMeanScalePrecision}) where {𝔽}
    return (M.dims[1] +1, )
end