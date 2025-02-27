"""
    get_natural_manifold_base(::Type{ExponentialFamily.GammaInverse}, ::Tuple{}, conditioner=nothing)

Get the natural manifold base for the `ExponentialFamily.GammaInverse` distribution.
"""
function get_natural_manifold_base(::Type{ExponentialFamily.GammaInverse}, ::Tuple{}, conditioner=nothing)
    return ProductManifold(
        PositiveVectors(1), PositiveVectors(1)
    )
end

"""
    partition_point(::Type{ExponentialFamily.GammaInverse}, ::Tuple{}, p, conditioner=nothing)

Converts the `point` to a compatible representation for the natural manifold of type `ExponentialFamily.GammaInverse`.
"""
function partition_point(::Type{ExponentialFamily.GammaInverse}, ::Tuple{}, p, conditioner=nothing)
    # For GammaInverse, we need to transform the parameters to work with Euclidean space
    # The natural parameters are in negative space, so we negate them
    p1 = -view(p, 1:1) .- 1
    p2 = -view(p, 2:2)
    return ArrayPartition(p1, p2)
end

"""
    transform_back!(p, ::NaturalParametersManifold{ExponentialFamily.GammaInverse}, q)

Transforms the `q` to a compatible representation for the exponential family distribution of type `ExponentialFamily.GammaInverse`.
"""
function transform_back!(p, ::NaturalParametersManifold{ℝ, ExponentialFamily.GammaInverse}, q)
    p[1:1] .= -(view(q, 1:1) .+ 1)
    p[2:2] .= -view(q, 2:2)
    return p
end