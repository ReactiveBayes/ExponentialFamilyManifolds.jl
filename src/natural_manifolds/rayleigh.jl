
"""
    get_natural_manifold_base(::Type{Rayleigh}, ::Tuple{}, conditioner=nothing)

Get the natural manifold base for the `Rayleigh` distribution.
"""
function get_natural_manifold_base(::Type{Rayleigh}, ::Tuple{}, conditioner=nothing)
    return PositiveVectors(1)
end

"""
    partition_point(::Type{Rayleigh}, ::Tuple{}, p, conditioner=nothing)

Converts the `point` to a compatible representation for the natural manifold of type `Laplace`.
"""
function partition_point(::Type{Rayleigh}, ::Tuple{}, p, conditioner=nothing)
    return -p
end

function transform_back!(p, ::NaturalParametersManifold{Rayleigh}, q)
    p .= -q
    return p
end