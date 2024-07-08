"""
    get_natural_manifold_base(::Type{Binomial}, ::Tuple{}, conditioner=nothing)

Get the natural manifold base for the `Binomial` distribution.
"""
function get_natural_manifold_base(::Type{Binomial}, ::Tuple{}, conditioner=nothing)
    @assert conditioner >= 0 "Conditioner $(conditioner) should be positive"
    return Euclidean(1)
end

"""
    partition_point(::Type{Binomial}, ::Tuple{}, p, conditioner=nothing)

Converts the `point` to a compatible representation for the natural manifold of type `Binomial`.
"""
function partition_point(::Type{Binomial}, ::Tuple{}, p, conditioner=nothing)
    @assert conditioner >= 0 "Conditioner $(conditioner) should be positive"
    return p
end