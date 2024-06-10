function get_natural_manifold_base(
    ::Type{Dirichlet},
    dims::Tuple{Int},
    conditioner = nothing,
)
    # `ProductManifold` here is important to treat the `PowerManifold` as a vector, and not matrix
    return ProductManifold(PowerManifold(ShiftedPositiveNumbers(static(-1)), first(dims)))
end

function partition_point(::Type{Dirichlet}, dims::Tuple{Int}, p, conditioner = nothing)
    return p
end