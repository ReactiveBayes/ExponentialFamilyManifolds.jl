"""
    get_natural_manifold_base(::Type{Weibull}, ::Tuple{}, conditioner=nothing)

Get the natural manifold base for the `Weibull` distribution.
"""
function get_natural_manifold_base(::Type{Weibull}, ::Tuple{}, conditioner=nothing)
    if conditioner === nothing
        throw(
            ArgumentError(
                "get_natural_manifold_base(::Type{Weibull},..., conditioner): `conditioner` was left as `nothing`. Please provide a non-negative numeric conditioner (e.g. 0.0 or 1.0).",
            ),
        )
    end
    if !(conditioner isa Number)
        throw(
            ArgumentError(
                "get_natural_manifold_base(::Type{Weibull},..., conditioner): `conditioner` must be a Number, got $(typeof(conditioner)).",
            ),
        )
    end
    if conditioner < 0
        throw(
            ArgumentError(
                "get_natural_manifold_base(::Type{Weibull},..., conditioner): `conditioner` must be >= 0, got $(conditioner).",
            ),
        )
    end
    return ProductManifold(PositiveVectors(1))
end

"""
    partition_point(::Type{Weibull}, ::Tuple{}, p, conditioner=nothing)

Converts the `point` to a compatible representation for the natural manifold of type `Weibull`.
"""
function partition_point(::Type{Weibull}, ::Tuple{}, p, conditioner=nothing)
    if conditioner === nothing
        throw(
            ArgumentError(
                "partition_point(::Type{Weibull},..., conditioner): `conditioner` was left as `nothing`. Please provide a non-negative numeric conditioner.",
            ),
        )
    end
    if !(conditioner isa Number)
        throw(
            ArgumentError(
                "partition_point(::Type{Weibull},..., conditioner): `conditioner` must be a Number, got $(typeof(conditioner)).",
            ),
        )
    end
    if conditioner < 0
        throw(
            ArgumentError(
                "partition_point(::Type{Weibull},..., conditioner): `conditioner` must be >= 0, got $(conditioner).",
            ),
        )
    end
    return ArrayPartition(-p)
end

"""
    transform_back!(p, ::NaturalParametersManifold{Weibull}, q)

Transforms the `q` to a compatible representation for the exponential family distribution of type `Weibull`.
"""
function transform_back!(p, M::NaturalParametersManifold{ℝ,Weibull}, q)
    p .= -q
    conditioner = getconditioner(M)
    @assert conditioner > 0 "Conditioner $(conditioner) should be positive"
    return p
end
