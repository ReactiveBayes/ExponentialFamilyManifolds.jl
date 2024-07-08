"""
    get_natural_manifold_base(::Type{NegativeBinomial}, ::Tuple{}, conditioner=nothing)

Get the natural manifold base for the `NegativeBinomial` distribution.
"""
function get_natural_manifold_base(::Type{NegativeBinomial}, ::Tuple{}, conditioner=nothing)
    @assert conditioner >= 0 "Conditioner should be non-negative"
    return Euclidean(1)
end

"""
    partition_point(::Type{NegativeBinomial}, ::Tuple{}, p, conditioner=nothing)

Converts the `point` to a compatible representation for the natural manifold of type `NegativeBinomial`.
"""
function partition_point(::Type{NegativeBinomial}, ::Tuple{}, p, conditioner=nothing)
    @assert conditioner >= 0 "Conditioner should be negative"
    return p
end