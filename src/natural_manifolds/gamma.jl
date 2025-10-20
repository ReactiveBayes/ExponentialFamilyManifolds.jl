
"""
    get_natural_manifold_base(::Type{Gamma}, ::Tuple{}, conditioner=nothing)

Get the natural manifold base for the `Gamma` distribution.
"""
function get_natural_manifold_base(::Type{Gamma}, ::Tuple{}, conditioner=nothing)
    return ProductManifold(PositiveVectors(1), PositiveVectors(1))
end

"""
    partition_point(::Type{Gamma}, ::Tuple{}, p, conditioner=nothing)

Converts the `point` to a compatible representation for the natural manifold of type `Gamma`.
"""
function partition_point(::Type{Gamma}, ::Tuple{}, p, conditioner=nothing)
    return ArrayPartition(view(p, 1:1) .+ 1, -view(p, 2:2))
end

"""
    transform_back!(p, M::NaturalParametersManifold{ℝ, Gamma}, q)

Transforms the `q` to a compatible representation for the exponential family distribution of type `Gamma`.
"""
function transform_back!(p, ::NaturalParametersManifold{ℝ,Gamma}, q)
    p[1:1] .= view(q, 1:1) .- 1
    p[2:2] .= -view(q, 2:2)
    return p
end
