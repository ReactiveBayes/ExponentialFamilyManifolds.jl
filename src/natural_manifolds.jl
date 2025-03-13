using ManifoldsBase, Manifolds, Static, RecursiveArrayTools, Random, ExponentialFamily

import ExponentialFamily: exponential_family_typetag

"""
    NaturalParametersManifold(::Type{T}, dims, base, conditioner)

The manifold for the natural parameters of the distribution of type `T` with dimensions `dims`.
An internal structure, use `get_natural_manifold` to create an instance of a manifold for the natural parameters of distribution of type `T`.
"""
struct NaturalParametersManifold{𝔽,T,D,M,C} <: AbstractDecoratorManifold{𝔽}
    dims::D
    base::M
    conditioner::C
end

getdims(M::NaturalParametersManifold) = M.dims
getbase(M::NaturalParametersManifold) = M.base
getconditioner(M::NaturalParametersManifold) = M.conditioner

# The `NaturalParametersManifold` simply adds extra properties to the `base` and 
# acts as a "decorator"
@inline ManifoldsBase.active_traits(f::F, ::NaturalParametersManifold, ::Any...) where {F} =
    ManifoldsBase.IsExplicitDecorator()
@inline ManifoldsBase.decorated_manifold(M::NaturalParametersManifold) = M.base

function ExponentialFamily.exponential_family_typetag(
    ::NaturalParametersManifold{𝔽,T}
) where {𝔽,T}
    return T
end

function NaturalParametersManifold(
    ::Type{T}, dims::D, base::M, conditioner::C=nothing
) where {T,𝔽,D,M<:AbstractManifold{𝔽},C}
    return NaturalParametersManifold{𝔽,T,D,M,C}(dims, base, conditioner)
end

"""
    get_natural_manifold(::Type{T}, dims, conditioner = nothing)

The function returns a corresponding manifold for the natural parameters of distribution of type `T`.
Optionally accepts the conditioner, which is set to `nothing` by default. Use empty tuple `()` for univariate distributions. 

```jldoctest
julia> using ExponentialFamily, ExponentialFamilyManifolds

julia> ExponentialFamilyManifolds.get_natural_manifold(Beta, ()) isa ExponentialFamilyManifolds.NaturalParametersManifold
true

julia> ExponentialFamilyManifolds.get_natural_manifold(MvNormalMeanCovariance, (3, )) isa ExponentialFamilyManifolds.NaturalParametersManifold
true
```
"""
function get_natural_manifold(::Type{T}, dims, conditioner=nothing) where {T}
    return NaturalParametersManifold(
        T, dims, get_natural_manifold_base(T, dims, conditioner), conditioner
    )
end

"""
    get_natural_manifold_base(M::NaturalParametersManifold)
    get_natural_manifold_base(::Type{T}, dims, conditioner = nothing)

Returns `base` manifold for the distribution of type `T` of dimension `dims`.
Optionally accepts the conditioner, which is set to `nothing` by default.
"""
function get_natural_manifold_base(M::NaturalParametersManifold) 
    return getbase(M)
end

"""
    partition_point(M::NaturalParametersManifold, p)
    partition_point(::Type{T}, dims, point, conditioner=nothing)

Converts the `point` to a compatible representation for the natural manifold `M` of type `T`.
"""
function partition_point(M::NaturalParametersManifold, p)
    return partition_point(exponential_family_typetag(M), getdims(M), p, getconditioner(M))
end

"""
    transform_back!(p, ::NaturalParametersManifold, q)

Transforms the `q` to a compatible representation for the exponential family distribution of type `T`.
"""
function transform_back!(_, ::NaturalParametersManifold{𝔽,T}, q) where {𝔽,T}
    error("You need to implement `transform_back!` for your specific $T of the exponential family distribution")
end

function Base.convert(
    ::Type{ExponentialFamilyDistribution}, M::NaturalParametersManifold, p
)   
    return ExponentialFamilyDistribution(
        exponential_family_typetag(M), transform_back(M, p), getconditioner(M), nothing
    )
end
