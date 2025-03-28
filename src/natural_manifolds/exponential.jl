"""
    get_natural_manifold_base(::Type{Exponential}, ::Tuple{}, conditioner=nothing)

Get the natural manifold base for the `Exponential` distribution.
"""
function get_natural_manifold_base(::Type{Exponential}, ::Tuple{}, conditioner=nothing)
    return ProductManifold(PositiveVectors(1))
end

"""
    partition_point(::Type{Exponential}, ::Tuple{}, p, conditioner=nothing)

Converts the `point` to a compatible representation for the natural manifold of type `Exponential`.
"""
function partition_point(::Type{Exponential}, ::Tuple{}, p, conditioner=nothing)
    return ArrayPartition(-p)
end

"""
    transform_back!(p, ::NaturalParametersManifold{Exponential}, q)

Transforms the `q` to a compatible representation for the exponential family distribution of type `Exponential`.
"""
function transform_back!(p, ::NaturalParametersManifold{ℝ,Exponential}, q)
    p .= -q
    return p
end
