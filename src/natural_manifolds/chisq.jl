
"""
    get_natural_manifold_base(::Type{Chisq}, ::Tuple{}, conditioner=nothing)

Get the natural manifold base for the `Chisq` distribution.
"""
function get_natural_manifold_base(::Type{Chisq}, ::Tuple{}, conditioner=nothing)
    return ShiftedPositiveNumbers(static(-1))
end

"""
    partition_point(::Type{Chisq}, ::Tuple{}, p, conditioner=nothing)

Converts the `point` to a compatible representation for the natural manifold of type `Chisq`.
"""
function partition_point(::Type{Chisq}, ::Tuple{}, p, conditioner=nothing)
    return p
end