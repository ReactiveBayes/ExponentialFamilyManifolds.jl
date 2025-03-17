using ManifoldsBase, Manifolds, Static, RecursiveArrayTools, Random, ExponentialFamily

import ExponentialFamily: exponential_family_typetag

"""
    NaturalParametersManifold(::Type{T}, dims, base, conditioner)

The manifold for the natural parameters of the distribution of type `T` with dimensions `dims`.
An internal structure, use `get_natural_manifold` to create an instance of a manifold for the natural parameters of distribution of type `T`.
"""
struct NaturalParametersManifold{𝔽,T,D,M,C,R} <: AbstractDecoratorManifold{𝔽}
    dims::D
    base::M
    conditioner::C
    retraction::R
end

getdims(M::NaturalParametersManifold) = M.dims
getbase(M::NaturalParametersManifold) = M.base
getconditioner(M::NaturalParametersManifold) = M.conditioner
getretraction(M::NaturalParametersManifold) = M.retraction

# The `NaturalParametersManifold` simply adds extra properties to the `base` and 
# acts as a "decorator"
@inline function ManifoldsBase.active_traits(f::F, ::NaturalParametersManifold, args...) where {F}
    # Don't delegate retraction-related methods
    if f in (ManifoldsBase.retract, ManifoldsBase.retract!, 
             ManifoldsBase.retract_fused, ManifoldsBase.retract_fused!)
        return ManifoldsBase.EmptyTrait()
    else
        return ManifoldsBase.IsExplicitDecorator()
    end
end

@inline ManifoldsBase.decorated_manifold(M::NaturalParametersManifold) = M.base

function ExponentialFamily.exponential_family_typetag(
    ::NaturalParametersManifold{𝔽,T}
) where {𝔽,T}
    return T
end

function NaturalParametersManifold(
    ::Type{T}, dims::D, base::M, conditioner::C=nothing, retraction::R=nothing
) where {T,𝔽,D,M<:AbstractManifold{𝔽},C,R}
    return NaturalParametersManifold{𝔽,T,D,M,C,R}(dims, base, conditioner, retraction)
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
function get_natural_manifold(::Type{T}, dims, conditioner=nothing, retraction = nothing) where {T}
    return NaturalParametersManifold(
        T, dims, get_natural_manifold_base(T, dims, conditioner), conditioner, retraction
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

function Base.convert(
    ::Type{ExponentialFamilyDistribution}, M::NaturalParametersManifold, p
)
    # The extra `nothing` at the end bypasses the check that the `p` is a 
    # valid vector of parameters
    return ExponentialFamilyDistribution(
        exponential_family_typetag(M), p, getconditioner(M), nothing
    )
end

struct ChartNOrderRetraction{Order, E} <: AbstractRetractionMethod
    extra::E
end

function ChartNOrderRetraction{O}() where {O}
    return ChartNOrderRetraction{O, Nothing}(nothing)
end

const FirstOrderRetraction = ChartNOrderRetraction{1}

ManifoldsBase.default_retraction_method(::NaturalParametersManifold{𝔽,TD,D,M,C,Nothing}, ::Type{T}) where {𝔽, T,TD, D, M, C} = FirstOrderRetraction()

ManifoldsBase.default_retraction_method(M::NaturalParametersManifold{𝔽,TD,D,BM,C,R}, ::Type{T}) where {𝔽, T,TD, D, BM, C, R} = getretraction(M)

function ManifoldsBase.retract_fused!(::NaturalParametersManifold, q, p, X, t::Number, method::FirstOrderRetraction)
    q .= p .+ t .* X
    return q
end

function ManifoldsBase.retract!(M::NaturalParametersManifold, q, p, X, method::FirstOrderRetraction)
    return ManifoldsBase.retract_fused!(M, q, p, X, one(eltype(X)), method)
end


