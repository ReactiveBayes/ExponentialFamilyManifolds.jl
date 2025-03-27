"""
    get_natural_manifold_base(::Type{Poisson}, ::Tuple{}, conditioner=nothing)

Get the natural manifold base for the `Poisson` distribution.
"""
function get_natural_manifold_base(::Type{Poisson}, ::Tuple{}, conditioner=nothing)
    return Euclidean(1)
end

"""
    partition_point(::Type{Poisson}, ::Tuple{}, p, conditioner=nothing)

Converts the `point` to a compatible representation for the natural manifold of type `Poisson`.
"""
function partition_point(::Type{Poisson}, ::Tuple{}, p, conditioner=nothing)
    return p
end

"""
    transform_back!(p, ::NaturalParametersManifold{ℝ, Poisson}, q)

Transforms the `q` to a compatible representation for the exponential family distribution of type `Bernoulli`.
"""
function transform_back!(p, ::NaturalParametersManifold{ℝ,Poisson}, q)
    return p .= q
end
