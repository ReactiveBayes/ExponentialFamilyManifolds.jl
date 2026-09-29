"""
    get_natural_manifold_base(::Type{Binomial}, ::Tuple{}, conditioner=nothing)

Get the natural manifold base for the `Binomial` distribution.
"""
function get_natural_manifold_base(::Type{Binomial}, ::Tuple{}, conditioner=nothing)
    if conditioner === nothing
        throw(
            ArgumentError(
                "get_natural_manifold_base(::Type{Binomial},..., conditioner): `conditioner` was left as `nothing`. Please provide a non-negative numeric conditioner (e.g. 0.0 or 1.0).",
            ),
        )
    end
    if !(conditioner isa Number)
        throw(
            ArgumentError(
                "get_natural_manifold_base(::Type{Binomial},..., conditioner): `conditioner` must be a Number, got $(typeof(conditioner)).",
            ),
        )
    end
    if conditioner < 0
        throw(
            ArgumentError(
                "get_natural_manifold_base(::Type{Binomial},..., conditioner): `conditioner` must be >= 0, got $(conditioner).",
            ),
        )
    end
    return Euclidean(1)
end

"""
    partition_point(::Type{Binomial}, ::Tuple{}, p, conditioner=nothing)

Converts the `point` to a compatible representation for the natural manifold of type `Binomial`.
"""
function partition_point(::Type{Binomial}, ::Tuple{}, p, conditioner=nothing)
    if conditioner === nothing
        throw(
            ArgumentError(
                "partition_point(::Type{Binomial},..., conditioner): `conditioner` was left as `nothing`. Please provide a non-negative numeric conditioner.",
            ),
        )
    end
    if !(conditioner isa Number)
        throw(
            ArgumentError(
                "partition_point(::Type{Binomial},..., conditioner): `conditioner` must be a Number, got $(typeof(conditioner)).",
            ),
        )
    end
    if conditioner < 0
        throw(
            ArgumentError(
                "partition_point(::Type{Binomial},..., conditioner): `conditioner` must be >= 0, got $(conditioner).",
            ),
        )
    end
    return p
end

"""
    transform_back!(p, ::NaturalParametersManifold{Binomial}, q)

Transforms the `q` to a compatible representation for the exponential family distribution of type `Binomial`.
"""
function transform_back!(p, ::NaturalParametersManifold{ℝ,Binomial}, q)
    p .= q
    return p
end
