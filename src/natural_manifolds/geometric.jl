
"""
    get_natural_manifold_base(::Type{Geometric}, ::Tuple{}, conditioner=nothing)

Get the natural manifold base for the `Geometric` distribution.
"""
function get_natural_manifold_base(::Type{Geometric}, ::Tuple{}, conditioner=nothing)
    return PositiveVectors(1)
end

"""
    partition_point(::Type{Geometric}, ::Tuple{}, p, conditioner=nothing)

Converts the `point` to a compatible representation for the natural manifold of type `Geometric`.
"""
function partition_point(::Type{Geometric}, ::Tuple{}, p, conditioner=nothing)
    return -p
end

"""
    transform_back!(p, ::NaturalParametersManifold{Geometric}, q)

Transforms the `q` to a compatible representation for the exponential family distribution of type `Geometric`.
"""
function transform_back!(p, ::NaturalParametersManifold{ℝ, Geometric}, q)
    p .= -q
    return p
end
