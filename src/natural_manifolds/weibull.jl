"""
    get_natural_manifold_base(::Type{Weibull}, ::Tuple{}, conditioner=nothing)

Get the natural manifold base for the `Weibull` distribution.
"""
function get_natural_manifold_base(::Type{Weibull}, ::Tuple{}, conditioner=nothing)
    @assert conditioner > 0 "Conditioner $(conditioner) should be positive"
    return PositiveVectors(1)
end

"""
    partition_point(::Type{Weibull}, ::Tuple{}, p, conditioner=nothing)

Converts the `point` to a compatible representation for the natural manifold of type `Weibull`.
"""
function partition_point(::Type{Weibull}, ::Tuple{}, p, conditioner=nothing)
    @assert conditioner > 0  "Conditioner $(conditioner) should be positive"
    return -p
end

"""
    transform_back!(p, ::NaturalParametersManifold{Weibull}, q)

Transforms the `q` to a compatible representation for the exponential family distribution of type `Weibull`.
"""
function transform_back!(p, M::NaturalParametersManifold{ℝ, Weibull}, q)
    p .= -q
    conditioner = getconditioner(M)
    @assert conditioner > 0 "Conditioner $(conditioner) should be positive"
    return p
end
