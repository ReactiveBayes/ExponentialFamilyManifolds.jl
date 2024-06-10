using ManifoldsBase, Manifolds, Static, RecursiveArrayTools, Random, ExponentialFamily

import ExponentialFamily: exponential_family_typetag

struct NaturalParametersManifold{𝔽,T,D,M,C} <: AbstractDecoratorManifold{𝔽}
    dims::D
    base::M
    conditioner::C
end

getbase(M::NaturalParametersManifold) = M.base
getconditioner(M::NaturalParametersManifold) = M.conditioner

# The `NaturalParametersManifold` simply adds extra properties to the `base` and 
# acts as a "decorator"
@inline ManifoldsBase.active_traits(f::F, ::NaturalParametersManifold, ::Any...) where {F} =
    ManifoldsBase.IsExplicitDecorator()
@inline ManifoldsBase.decorated_manifold(M::NaturalParametersManifold) = M.base

ExponentialFamily.exponential_family_typetag(::NaturalParametersManifold{𝔽,T}) where {𝔽,T} =
    T

function NaturalParametersManifold(
    ::Type{T},
    dims::D,
    base::M,
    conditioner::C = nothing,
) where {T,𝔽,D,M<:AbstractManifold{𝔽},C}
    return NaturalParametersManifold{𝔽,T,D,M,C}(dims, base, conditioner)
end

"""
    get_natural_manifold(::Type{T}, conditioner = nothing)

The function returns a corresponding manifold for the natural parameters of distribution of type `T`.
Optionally accepts the conditioner, which is set to `nothing` by default.
"""
function get_natural_manifold(::Type{T}, dims, conditioner = nothing) where {T}
    return NaturalParametersManifold(
        T,
        dims,
        get_natural_manifold_base(T, dims, conditioner),
        conditioner,
    )
end

function get_natural_manifold(ef::ExponentialFamilyDistribution)
    return get_natural_manifold(exponential_family_typetag(ef), getconditioner(ef))
end

"""
    get_natural_manifold_base(::Type{T}, conditioner = nothing)

Returns `base` manifold for the distribution of type `T`.
"""
function get_natural_manifold_base end

"""
    partition_point(M::NaturalParametersManifold, point)
    partition_point(::Type{T}, conditioner, point)

Converts `point` to a compatible representation for the natural manifold `M` of type `T`.
"""
function partition_point(M::NaturalParametersManifold, dims, p)
    return partition_point(exponential_family_typetag(M), dims, p, getconditioner(M))
end

function Base.convert(
    ::Type{ExponentialFamilyDistribution},
    M::NaturalParametersManifold,
    p,
)
    # The extra `nothing` at the end bypasses the check that the `p` is a 
    # valid vector of parameters
    return ExponentialFamilyDistribution(
        exponential_family_typetag(M),
        p,
        getconditioner(M),
        nothing,
    )
end

