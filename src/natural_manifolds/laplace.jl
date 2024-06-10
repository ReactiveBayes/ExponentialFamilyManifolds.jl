
"""
    get_natural_manifold_base(::Type{Laplace}, ::Tuple{}, conditioner=nothing)

Get the natural manifold base for the `Laplace` distribution.
"""
function get_natural_manifold_base(::Type{Laplace}, ::Tuple{}, conditioner=nothing)
    return ShiftedNegativeNumbers(static(0))
end

"""
    partition_point(::Type{Laplace}, ::Tuple{}, p, conditioner=nothing)

Converts the `point` to a compatible representation for the natural manifold of type `Laplace`.
"""
function partition_point(::Type{Laplace}, ::Tuple{}, p, conditioner=nothing)
    return p
end