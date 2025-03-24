"""
    get_natural_manifold_base(::Type{Bernoulli}, ::Tuple{}, conditioner=nothing)

Get the natural manifold base for the `Bernoulli` distribution.
"""
function get_natural_manifold_base(::Type{Bernoulli}, ::Tuple{}, conditioner=nothing)
    return Euclidean(1)
end

"""
    partition_point(::Type{Bernoulli}, ::Tuple{}, p, conditioner=nothing)

Converts the `point` to a compatible representation for the natural manifold of type `Bernoulli`.
"""
function partition_point(::Type{Bernoulli}, ::Tuple{}, p, conditioner=nothing)
    return p
end

"""
    transform_back!(p, M::NaturalParametersManifold{ℝ, Bernoulli}, q)

Transforms the natural parameter `q` back to the probability parameter for the Bernoulli distribution.
"""
function transform_back!(p, ::NaturalParametersManifold{ℝ, Bernoulli}, q)
    p .= q
    return p
end
