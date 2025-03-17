"""
    get_natural_manifold_base(::Type{Beta}, ::Tuple{}, conditioner=nothing)

Get the natural manifold base for the `Beta` distribution.
"""
function get_natural_manifold_base(::Type{Beta}, ::Tuple{}, conditioner=nothing)
    return ProductManifold(
        ShiftedPositiveNumbers(static(-1)), ShiftedPositiveNumbers(static(-1))
    )
end

"""
    partition_point(::Type{Beta}, ::Tuple{}, p, conditioner=nothing)

Converts the `point` to a compatible representation for the natural manifold of type `Beta`.
"""
function partition_point(::Type{Beta}, ::Tuple{}, p, conditioner=nothing)
    return ArrayPartition(view(p, 1:1), view(p, 2:2))
end
