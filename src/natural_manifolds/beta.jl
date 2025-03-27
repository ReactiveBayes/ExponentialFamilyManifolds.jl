"""
    get_natural_manifold_base(::Type{Beta}, ::Tuple{}, conditioner=nothing)

Get the natural manifold base for the `Beta` distribution.
"""
function get_natural_manifold_base(::Type{Beta}, ::Tuple{}, conditioner=nothing)
    return ProductManifold(PositiveVectors(1), PositiveVectors(1))
end

"""
    partition_point(::Type{Beta}, ::Tuple{}, p, conditioner=nothing)

Converts the `point` to a compatible representation for the natural manifold of type `Beta`.
"""
function partition_point(::Type{Beta}, ::Tuple{}, p, conditioner=nothing)
    return ArrayPartition(view(p, 1:1), view(p, 2:2)) .+ 1
end

"""
    transform_back!(p, ::NaturalParametersManifold{Beta}, q)

Transforms the `q` to a compatible representation for the exponential family distribution of type `Beta`.
"""
function transform_back!(p, ::NaturalParametersManifold{ℝ,Beta}, q)
    p .= q .- 1
    return p
end
