function get_natural_manifold_base(::Type{Laplace}, ::Tuple{}, conditioner = nothing)
    return ShiftedNegativeNumbers(static(0))
end

function partition_point(::Type{Laplace}, ::Tuple{}, p, conditioner = nothing)
    return p
end