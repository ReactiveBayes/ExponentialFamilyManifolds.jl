
"""
    get_natural_manifold_base(::Type{Categorical}, dims::Tuple{Int}, conditioner=nothing)

Get the natural manifold base for the `Categorical` distribution.
"""
function get_natural_manifold_base(::Type{Categorical}, ::Tuple{}, conditioner=nothing)
    return ProductManifold(
        Euclidean(conditioner-1), SinglePointManifold(0)
    )
end

"""
    partition_point(::Type{Categorical}, dims::Tuple{Int}, p, conditioner=nothing)

Converts the `point` to a compatible representation for the natural manifold of type `Categorical`.
"""
function partition_point(::Type{Categorical}, ::Tuple{}, p, conditioner=nothing)
    # See comment in `get_natural_manifold_base` for `Categorical`
    return ArrayPartition(p[1:end-1], p[end])
end