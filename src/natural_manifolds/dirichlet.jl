
"""
    get_natural_manifold_base(::Type{Dirichlet}, dims::Tuple{Int}, conditioner=nothing)

Get the natural manifold base for the `Dirichlet` distribution.
"""
function get_natural_manifold_base(::Type{Dirichlet}, dims::Tuple{Int}, conditioner=nothing)
    # `PowerManifold` does treat the vector as a matrix with one row
    # In the `parition_point` we transpose the vector and use `ArrayPartition` for `ProductManifold`
    return ProductManifold(PowerManifold(PositiveVectors(1), first(dims)))
end

"""
    partition_point(::Type{Dirichlet}, dims::Tuple{Int}, p, conditioner=nothing)

Converts the `point` to a compatible representation for the natural manifold of type `Dirichlet`.
"""
function partition_point(::Type{Dirichlet}, dims::Tuple{Int}, p, conditioner=nothing)
    # See comment in `get_natural_manifold_base` for `Dirichlet`
    return ArrayPartition(p') .+ 1
end

"""
    transform_back!(p, M::NaturalParametersManifold{ℝ, Dirichlet}, q)

Transforms the `q` to a compatible representation for the exponential family distribution of type `Dirichlet`.
"""
function transform_back!(p, ::NaturalParametersManifold{ℝ, Dirichlet}, q)
    p .= q .- 1
    return p
end
