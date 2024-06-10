function get_natural_manifold_base(::Type{Bernoulli}, ::Tuple{}, conditioner = nothing)
    return Euclidean(1)
end

function partition_point(::Type{Bernoulli}, ::Tuple{}, p, conditioner = nothing)
    return p
end