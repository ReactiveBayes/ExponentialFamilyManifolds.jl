
"""
    get_natural_manifold_base(::Type{Dirichlet}, dims::Tuple{Int}, conditioner=nothing)

Get the natural manifold base for the `Dirichlet` distribution.
"""
function get_natural_manifold_base(::Type{Dirichlet}, dims::Tuple{Int}, conditioner=nothing)
    # `PowerManifold` does treat the vector as a matrix with one row
    # In the `parition_point` we transpose the vector and use `ArrayPartition` for `ProductManifold`
    return ProductManifold(PowerManifold(ShiftedPositiveNumbers(static(-1)), first(dims)))
end

"""
    partition_point(::Type{Dirichlet}, dims::Tuple{Int}, p, conditioner=nothing)

Converts the `point` to a compatible representation for the natural manifold of type `Dirichlet`.
"""
function partition_point(::Type{Dirichlet}, dims::Tuple{Int}, p, conditioner=nothing)
    # See comment in `get_natural_manifold_base` for `Dirichlet`
    return ArrayPartition(p')
end
