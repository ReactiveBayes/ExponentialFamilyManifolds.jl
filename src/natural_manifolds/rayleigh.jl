
"""
    get_natural_manifold_base(::Type{Rayleigh}, ::Tuple{}, conditioner=nothing)

Get the natural manifold base for the `Rayleigh` distribution.
"""
function get_natural_manifold_base(::Type{Rayleigh}, ::Tuple{}, conditioner=nothing)
    return ProductManifold(PositiveVectors(1))
end

"""
    partition_point(::Type{Rayleigh}, ::Tuple{}, p, conditioner=nothing)

Converts the `point` to a compatible representation for the natural manifold of type `Laplace`.
"""
function partition_point(::Type{Rayleigh}, ::Tuple{}, p, conditioner=nothing)
    return ArrayPartition(-p)
end

"""
    transform_back!(p, ::NaturalParametersManifold{Rayleigh}, q)

Transforms the `q` to a compatible representation for the exponential family distribution of type `Rayleigh`.
"""
function transform_back!(p, ::NaturalParametersManifold{ℝ,Rayleigh}, q)
    p .= -q
    return p
end
