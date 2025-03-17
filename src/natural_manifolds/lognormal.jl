"""
    get_natural_manifold_base(::Type{LogNormal}, ::Tuple{}, conditioner=nothing)

Get the natural manifold base for the `LogNormal` distribution.
"""
function get_natural_manifold_base(::Type{LogNormal}, ::Tuple{}, conditioner=nothing)
    return ProductManifold(Euclidean(1), ShiftedNegativeNumbers(static(0)))
end

"""
    partition_point(::Type{LogNormal}, ::Tuple{}, p, conditioner=nothing)

Converts the `point` to a compatible representation for the natural manifold of type `LogNormal`.
"""
function partition_point(::Type{LogNormal}, ::Tuple{}, p, conditioner=nothing)
    return ArrayPartition(view(p, 1:1), view(p, 2:2))
end
