"""
    get_natural_manifold_base(::Type{Exponential}, ::Tuple{}, conditioner=nothing)

Get the natural manifold base for the `Exponential` distribution.
"""
function get_natural_manifold_base(::Type{Exponential}, ::Tuple{}, conditioner=nothing)
    return ShiftedNegativeNumbers(static(0))
end

"""
    partition_point(::Type{Exponential}, ::Tuple{}, p, conditioner=nothing)

Converts the `point` to a compatible representation for the natural manifold of type `Exponential`.
"""
function partition_point(::Type{Exponential}, ::Tuple{}, p, conditioner=nothing)
    return p
end
