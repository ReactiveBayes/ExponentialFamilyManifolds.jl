"""
    get_natural_manifold_base(::Type{WishartFast}, ::Tuple{Int}, conditioner=nothing)

Get the natural manifold base for the `WishartFast` distribution.
"""
function get_natural_manifold_base(
    ::Type{ExponentialFamily.WishartFast}, dims::Tuple{Int,Int}, conditioner=nothing
)
    k = first(dims)
    return ProductManifold(ShiftedPositiveNumbers(static(0)), SymmetricNegativeDefinite(k))
end

"""
    partition_point(::Type{WishartFast}, ::Tuple{Int}, p, conditioner=nothing)

Converts the `point` to a compatible representation for the natural manifold of type `WishartFast`.
"""
function partition_point(
    ::Type{ExponentialFamily.WishartFast}, dims::Tuple{Int,Int}, p, conditioner=nothing
)
    k = first(dims)
    return ArrayPartition(view(p, 1:1), reshape(view(p, 2:(1 + k^2)), (k, k)))
end

function getdefaultmetric(::Type{ExponentialFamily.WishartFast})
    return BaseMetric()
end