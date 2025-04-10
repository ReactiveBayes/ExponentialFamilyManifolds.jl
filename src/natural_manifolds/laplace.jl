
"""
    get_natural_manifold_base(::Type{Laplace}, ::Tuple{}, conditioner=nothing)

Get the natural manifold base for the `Laplace` distribution.
"""
function get_natural_manifold_base(::Type{Laplace}, ::Tuple{}, conditioner=nothing)
    return ProductManifold(PositiveVectors(1))
end

"""
    partition_point(::Type{Laplace}, ::Tuple{}, p, conditioner=nothing)

Converts the `point` to a compatible representation for the natural manifold of type `Laplace`.
"""
function partition_point(::Type{Laplace}, ::Tuple{}, p, conditioner=nothing)
    return ArrayPartition(-p)
end

"""
    transform_back!(p, ::NaturalParametersManifold{Laplace}, q)

Transforms the `q` to a compatible representation for the exponential family distribution of type `Laplace`.
"""
function transform_back!(p, ::NaturalParametersManifold{ℝ,Laplace}, q)
    p .= -q
    return p
end
