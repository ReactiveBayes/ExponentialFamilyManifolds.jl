"""
    get_natural_manifold_base(::Type{WishartFast}, ::Tuple{Int}, conditioner=nothing)

Get the natural manifold base for the `WishartFast` distribution.
"""
function get_natural_manifold_base(::Type{ExponentialFamily.WishartFast}, dims::Tuple{Int, Int}, conditioner=nothing)
    k = first(dims)
    return ProductManifold(PositiveVectors(1), SymmetricPositiveDefinite(k))
end

"""
    partition_point(::Type{WishartFast}, ::Tuple{Int}, p, conditioner=nothing)

Converts the `point` to a compatible representation for the natural manifold of type `WishartFast`.
"""
function partition_point(::Type{ExponentialFamily.WishartFast}, dims::Tuple{Int, Int}, p, conditioner=nothing)
    k = first(dims)
    return ArrayPartition(view(p, 1:1), -reshape(view(p, 2:(1 + k^2)), (k, k)))
end

"""
    transform_back!(p, M::NaturalParametersManifold{ℝ, WishartFast}, q)

Transforms the `q` to a compatible representation for the exponential family distribution of type `WishartFast`.
"""
function transform_back!(p, M::NaturalParametersManifold{ℝ, ExponentialFamily.WishartFast}, q)
    # k = first(getdims(M))
    # p[1:1] .= view(q, 1:1)
    p .= -q
    p[1:1] .= -view(q, 1:1)
    return p
end
