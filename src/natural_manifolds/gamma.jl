function get_natural_manifold_base(::Type{Gamma}, ::Tuple{}, conditioner=nothing)
    return ProductManifold(
        ShiftedPositiveNumbers(static(-1)), ShiftedNegativeNumbers(static(0))
    )
end

function partition_point(::Type{Gamma}, ::Tuple{}, p, conditioner=nothing)
    return ArrayPartition(view(p, 1:1), view(p, 2:2))
end