
"""
    get_natural_manifold_base(::Type{Geometric}, ::Tuple{}, conditioner=nothing)

Get the natural manifold base for the `Geometric` distribution.
"""
function get_natural_manifold_base(::Type{Geometric}, ::Tuple{}, conditioner=nothing)
    return ShiftedNegativeNumbers(static(0))
end

"""
    partition_point(::Type{Geometric}, ::Tuple{}, p, conditioner=nothing)

Converts the `point` to a compatible representation for the natural manifold of type `Geometric`.
"""
function partition_point(::Type{Geometric}, ::Tuple{}, p, conditioner=nothing)
    return p
end