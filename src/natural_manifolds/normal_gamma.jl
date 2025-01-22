"""
    get_natural_manifold_base(::Type{NormalGamma}, ::Tuple{}, conditioner=nothing)
Get the natural manifold base for the `NormalGamma` distribution.
"""
function get_natural_manifold_base(::Type{NormalGamma}, ::Tuple{Int}, conditioner=nothing)
    return NormalGammaNaturalManifold()
end

"""
    partition_point(::Type{NormalGamma}, ::Tuple{}, p, conditioner=nothing)
Converts the `point` to a compatible representation for the natural manifold of type `NormalGamma`.
"""
function partition_point(::Type{NormalGamma}, ::Tuple{Int}, p, conditioner=nothing)
    return p
end
