
"""
    get_natural_manifold_base(::Type{Pareto}, ::Tuple{}, conditioner=nothing)

Get the natural manifold base for the `Pareto` distribution.
"""
function get_natural_manifold_base(::Type{Pareto}, ::Tuple{}, conditioner=nothing)
    return ShiftedNegativeNumbers(static(-1))
end

"""
    partition_point(::Type{Pareto}, ::Tuple{}, p, conditioner=nothing)

Converts the `point` to a compatible representation for the natural manifold of type `Pareto`.
"""
function partition_point(::Type{Pareto}, ::Tuple{}, p, conditioner=nothing)
    return p
end
