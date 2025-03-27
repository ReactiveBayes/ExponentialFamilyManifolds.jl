
"""
    get_natural_manifold_base(::Type{Pareto}, ::Tuple{}, conditioner=nothing)

Get the natural manifold base for the `Pareto` distribution.
"""
function get_natural_manifold_base(::Type{Pareto}, ::Tuple{}, conditioner=nothing)
    return ProductManifold(PositiveVectors(1))
end

"""
    partition_point(::Type{Pareto}, ::Tuple{}, p, conditioner=nothing)

Converts the `point` to a compatible representation for the natural manifold of type `Pareto`.
"""
function partition_point(::Type{Pareto}, ::Tuple{}, p, conditioner=nothing)
    return ArrayPartition(-p .- 1)
end

"""
    transform_back!(p, ::NaturalParametersManifold{Pareto}, q)

Transforms the `q` to a compatible representation for the exponential family distribution of type `Pareto`.
"""
function transform_back!(p, ::NaturalParametersManifold{ℝ,Pareto}, q)
    p .= -(q .+ 1)
    return p
end
