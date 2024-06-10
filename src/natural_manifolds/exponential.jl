function get_natural_manifold_base(::Type{Exponential}, ::Tuple{}, conditioner = nothing)
    return ShiftedNegativeNumbers(static(0))
end

function partition_point(::Type{Exponential}, ::Tuple{}, p, conditioner = nothing)
    return p
end