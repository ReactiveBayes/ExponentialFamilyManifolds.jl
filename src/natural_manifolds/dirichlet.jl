
"""
    get_natural_manifold_base(::Type{Dirichlet}, dims::Tuple{Int}, conditioner=nothing)

Get the natural manifold base for the `Dirichlet` distribution.
"""
function get_natural_manifold_base(::Type{Dirichlet}, dims::Tuple{Int}, conditioner=nothing)
    # `PowerManifold` does treat the vector as a matrix with one row
    # In the `parition_point` we transpose the vector and use `ArrayPartition` for `ProductManifold`
    return ProductManifold(PowerManifold(PositiveVectors(1), first(dims)))
end

"""
    partition_point(::Type{Dirichlet}, dims::Tuple{Int}, p, conditioner=nothing)

Converts the `point` to a compatible representation for the natural manifold of type `Dirichlet`.
"""
function partition_point(::Type{Dirichlet}, dims::Tuple{Int}, p, conditioner=nothing)
    # See comment in `get_natural_manifold_base` for `Dirichlet`
    return ArrayPartition(p') .+ 1
end

"""
    transform_back!(p, M::NaturalParametersManifold{ℝ, Dirichlet}, q)

Transforms the `q` to a compatible representation for the exponential family distribution of type `Dirichlet`.
"""
function transform_back!(p, ::NaturalParametersManifold{ℝ,Dirichlet}, q)
    p .= q .- 1
    return p
end

# Friendly guards for the case when the user calls with an empty dims tuple `()`.
# Without these, Julia will fall through and raise a MethodError which is not helpful.

"""
    get_natural_manifold_base(::Type{Dirichlet}, ::Tuple{}, conditioner = nothing)

Guard method that throws a clear error when the Dirichlet manifold is requested
without the required dimension `K`. The Dirichlet distribution lives on the (K-1)
simplex and therefore needs the number of components `K` to construct the manifold.

Use `get_natural_manifold(Dirichlet, (K,))`, for example `get_natural_manifold(Dirichlet, (3,))`.
"""
function get_natural_manifold_base(::Type{Dirichlet}, ::Tuple{}, conditioner=nothing)
    throw(
        ArgumentError(
            "Dirichlet requires an explicit dimension `K`. " *
            "Call `ExponentialFamilyManifolds.get_natural_manifold(Dirichlet, (K,))` " *
            "for example `get_natural_manifold(Dirichlet, (3,))`.",
        ),
    )
end

"""
    partition_point(::Type{Dirichlet}, ::Tuple{}, p, conditioner=nothing)

Guard method for partition_point when dims == (). Provides a clearer error than the default
MethodError. Users should pass the dimension tuple `(K,)` to work with Dirichlet.
"""
function partition_point(::Type{Dirichlet}, ::Tuple{}, p, conditioner=nothing)
    throw(
        ArgumentError(
            "Dirichlet requires an explicit dimension `K` when partitioning points. " *
            "Call `get_natural_manifold(Dirichlet, (K,))` (e.g. `(3,)`) and retry.",
        ),
    )
end
