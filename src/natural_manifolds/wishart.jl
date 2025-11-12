"""
    get_natural_manifold_base(::Type{WishartFast}, ::Tuple{Int}, conditioner=nothing)

Get the natural manifold base for the `WishartFast` distribution.
"""
function get_natural_manifold_base(
    ::Type{ExponentialFamily.WishartFast}, dims::Tuple{Int,Int}, conditioner=nothing
)
    k = first(dims)
    return ProductManifold(PositiveVectors(1), SymmetricPositiveDefinite(k))
end

# Guard for missing dimension tuple in WishartFast
"""
    get_natural_manifold_base(::Type{ExponentialFamily.WishartFast}, ::Tuple{}, conditioner=nothing)

Raises a clear error when the WishartFast manifold is requested without the
required matrix dimension tuple `(K,K)`. The Wishart family is defined for
positive-definite matrices of size `K×K`.
"""
function get_natural_manifold_base(
    ::Type{ExponentialFamily.WishartFast}, ::Tuple{}, conditioner=nothing
)
    throw(
        ArgumentError(
            "WishartFast requires an explicit matrix dimension `(K,K)`. " *
            "Example: `get_natural_manifold(WishartFast, (3,3))`.",
        ),
    )
end

"""
    partition_point(::Type{WishartFast}, ::Tuple{Int}, p, conditioner=nothing)

Converts the `point` to a compatible representation for the natural manifold of type `WishartFast`.
"""
function partition_point(
    ::Type{ExponentialFamily.WishartFast}, dims::Tuple{Int,Int}, p, conditioner=nothing
)
    k = first(dims)
    return ArrayPartition(view(p, 1:1), -reshape(view(p, 2:(1 + k ^ 2)), (k, k)))
end

"""
    partition_point(::Type{ExponentialFamily.WishartFast}, ::Tuple{}, p, conditioner=nothing)

Guard method that throws an ArgumentError for missing dimension.
"""
function partition_point(
    ::Type{ExponentialFamily.WishartFast}, ::Tuple{}, p, conditioner=nothing
)
    throw(
        ArgumentError(
            "WishartFast requires an explicit matrix dimension `(K,K)` for partitioning points. " *
            "Example: `get_natural_manifold(WishartFast, (3,3))`.",
        ),
    )
end

"""
    transform_back!(p, M::NaturalParametersManifold{ℝ, WishartFast}, q)

Transforms the `q` to a compatible representation for the exponential family distribution of type `WishartFast`.
"""
function transform_back!(p, ::NaturalParametersManifold{ℝ,ExponentialFamily.WishartFast}, q)
    p .= -q
    p[1:1] .= view(q, 1:1)
    return p
end
