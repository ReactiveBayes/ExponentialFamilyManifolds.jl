
"""
    get_natural_manifold_base(::Type{Categorical}, dims::Tuple{Int}, conditioner=nothing)

Get the natural manifold base for the `Categorical` distribution.
"""
function get_natural_manifold_base(::Type{Categorical}, ::Tuple{}, conditioner=nothing)
    if conditioner === nothing
        throw(
            ArgumentError(
                "get_natural_manifold_base(::Type{Categorical},..., conditioner): `conditioner` was left as `nothing`. Please provide a non-negative numeric conditioner (e.g. 0.0 or 1.0).",
            ),
        )
    end
    if !(conditioner isa Number)
        throw(
            ArgumentError(
                "get_natural_manifold_base(::Type{Categorical},..., conditioner): `conditioner` must be a Number, got $(typeof(conditioner)).",
            ),
        )
    end
    if conditioner < 1
        throw(
            ArgumentError(
                "get_natural_manifold_base(::Type{Categorical},..., conditioner): `conditioner` must be >= 1, got $(conditioner).",
            ),
        )
    end
    return ProductManifold(Euclidean(conditioner - 1), SinglePointManifold([0.0]))
end

"""
    partition_point(::Type{Categorical}, dims::Tuple{Int}, p, conditioner=nothing)

Converts the `point` to a compatible representation for the natural manifold of type `Categorical`.
"""
function partition_point(::Type{Categorical}, ::Tuple{}, p, conditioner=nothing)
    if conditioner === nothing
        throw(
            ArgumentError(
                "partition_point(::Type{Categorical},..., conditioner): `conditioner` was left as `nothing`. Please provide a non-negative numeric conditioner.",
            ),
        )
    end
    if !(conditioner isa Number)
        throw(
            ArgumentError(
                "partition_point(::Type{Categorical},..., conditioner): `conditioner` must be a Number, got $(typeof(conditioner)).",
            ),
        )
    end
    if conditioner < 1
        throw(
            ArgumentError(
                "partition_point(::Type{Categorical},..., conditioner): `conditioner` must be >= 1, got $(conditioner).",
            ),
        )
    end
    return ArrayPartition(view(p, 1:(conditioner - 1)), view(p, conditioner:conditioner))
end

"""
    transform_back!(p, ::NaturalParametersManifold{Categorical}, q)

Transforms the `q` to a compatible representation for the exponential family distribution of type `Categorical`.
"""
function transform_back!(p, ::NaturalParametersManifold{ℝ,Categorical}, q)
    p .= q
    return p
end
