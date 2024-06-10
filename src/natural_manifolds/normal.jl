function get_natural_manifold_base(
    ::Type{NormalMeanVariance},
    ::Tuple{},
    conditioner = nothing,
)
    return ProductManifold(Euclidean(1), ShiftedNegativeNumbers(static(0)))
end

function partition_point(::Type{NormalMeanVariance}, ::Tuple{}, p, conditioner = nothing)
    return ArrayPartition(view(p, 1:1), view(p, 2:2))
end

function get_natural_manifold_base(
    ::Type{MvNormalMeanCovariance},
    dims::Tuple{Int},
    conditioner = nothing,
)
    k = first(dims)
    return ProductManifold(Euclidean(k), NegativeDefiniteMatrices(k))
end

function partition_point(
    ::Type{MvNormalMeanCovariance},
    dims::Tuple{Int},
    p,
    conditioner = nothing,
)
    k = first(dims)
    return ArrayPartition(view(p, 1:k), reshape(view(p, (k+1):(k+k^2)), (k, k)))
end