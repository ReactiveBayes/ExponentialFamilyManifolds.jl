"""
    get_natural_manifold_base(::Type{NormalGamma}, ::Tuple{}, conditioner=nothing)

Get the natural manifold base for the `NormalGamma` distribution.
"""
function get_natural_manifold_base(::Type{NormalGamma}, ::Tuple{Int}, conditioner=nothing)
    return ProductManifold(
        Euclidean(1), ShiftedNegativeNumbers(static(0)), ShiftedPositiveNumbers(static(-1/2)),ShiftedNegativeNumbers(static(0))
    )
end

"""
    partition_point(::Type{NormalGamma}, ::Tuple{}, p, conditioner=nothing)

Converts the `point` to a compatible representation for the natural manifold of type `NormalGamma`.
"""
function partition_point(::Type{NormalGamma}, ::Tuple{Int}, p, conditioner=nothing)
    return ArrayPartition(view(p, 1:1), view(p, 2:2), view(p, 3:3), view(p, 4:4))
end