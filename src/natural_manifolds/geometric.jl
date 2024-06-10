function get_natural_manifold_base(::Type{Geometric}, ::Tuple{}, conditioner=nothing)
    return ShiftedNegativeNumbers(static(0))
end

function partition_point(::Type{Geometric}, ::Tuple{}, p, conditioner=nothing)
    return p
end