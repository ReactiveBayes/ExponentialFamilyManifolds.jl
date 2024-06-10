function get_natural_manifold_base(::Type{Chisq}, ::Tuple{}, conditioner=nothing)
    return ShiftedPositiveNumbers(static(-1//2))
end

function partition_point(::Type{Chisq}, ::Tuple{}, p, conditioner=nothing)
    return p
end