"""
    get_natural_manifold_base(::Type{ExponentialFamily.GammaInverse}, ::Tuple{}, conditioner=nothing)

Get the natural manifold base for the `ExponentialFamily.GammaInverse` distribution.
"""
function get_natural_manifold_base(
    ::Type{ExponentialFamily.GammaInverse}, ::Tuple{}, conditioner=nothing
)
    return ProductManifold(
        ShiftedNegativeNumbers(static(-1)), ShiftedNegativeNumbers(static(0))
    )
end

"""
    partition_point(::Type{ExponentialFamily.GammaInverse}, ::Tuple{}, p, conditioner=nothing)

Converts the `point` to a compatible representation for the natural manifold of type `ExponentialFamily.GammaInverse`.
"""
function partition_point(
    ::Type{ExponentialFamily.GammaInverse}, ::Tuple{}, p, conditioner=nothing
)
    return ArrayPartition(view(p, 1:1), view(p, 2:2))
end
