using ManifoldsBase, Manifolds, Static, RecursiveArrayTools, Random, ExponentialFamily
using FastCholesky

import ExponentialFamily: exponential_family_typetag

"""
    FisherInformationMetric <: RiemannianMetric

Specifier that we need to use the Fisher information metric.
"""
struct FisherInformationMetric <: RiemannianMetric end

"""
    BaseMetric <: RiemannianMetric

Specifier that we need to use the metric from the base manifold.
"""
struct BaseMetric <: RiemannianMetric end

"""
    getdefaultmetric(::Type{T}) where {T}

Returns the default metric for the distribution of type `T`.
"""
function getdefaultmetric(::Type{T}) where {T}
    return FisherInformationMetric()
end

"""
    NaturalParametersManifold(::Type{T}, dims, base, conditioner)

The manifold for the natural parameters of the distribution of type `T` with dimensions `dims`.
An internal structure, use `get_natural_manifold` to create an instance of a manifold for the natural parameters of distribution of type `T`.
"""
struct NaturalParametersManifold{𝔽,T,D,M,C,R,MT} <: AbstractDecoratorManifold{𝔽}
    dims::D
    base::M
    conditioner::C
    retraction::R
    metric::MT
end

getdims(M::NaturalParametersManifold) = M.dims
getbase(M::NaturalParametersManifold) = M.base
getconditioner(M::NaturalParametersManifold) = M.conditioner
getretraction(M::NaturalParametersManifold) = M.retraction

# The `NaturalParametersManifold` simply adds extra properties to the `base` and 
# acts as a "decorator"
function select_skip_methods(::F, ::NaturalParametersManifold{𝔽,T,D,MB,C,R,BaseMetric}) where {F,𝔽,T,D,MB,C,R}
    return ManifoldsBase.IsExplicitDecorator()
end

function select_skip_methods(f::F, ::NaturalParametersManifold{𝔽,T,D,MB,C,R,FisherInformationMetric}) where {F,𝔽,T,D,MB,C,R}
    if f in (
        ManifoldsBase.retract,
        ManifoldsBase.retract!,
        ManifoldsBase.retract_fused,
        ManifoldsBase.retract_fused!,
        Manifolds.local_metric,
        Manifolds.local_metric_jacobian,
        Manifolds.default_retraction_method,
        Manifolds.get_basis_default,
        Manifolds.christoffel_symbols_second,
        Manifolds.christoffel_symbols_first
    )
        return ManifoldsBase.EmptyTrait()
    else
        return ManifoldsBase.IsExplicitDecorator()
    end
end

@inline function ManifoldsBase.active_traits(
    f::F, M::NaturalParametersManifold, args...
) where {F}
    return select_skip_methods(f, M)
end

@inline ManifoldsBase.decorated_manifold(M::NaturalParametersManifold) = M.base

function ExponentialFamily.exponential_family_typetag(
    ::NaturalParametersManifold{𝔽,T}
) where {𝔽,T}
    return T
end

function NaturalParametersManifold(
    ::Type{T},
    dims::D,
    base::M,
    conditioner::C=nothing,
    retraction::R=nothing,
    metric::MT=getdefaultmetric(T),
) where {T,𝔽,D,M<:AbstractManifold{𝔽},C,R,MT}
    return NaturalParametersManifold{𝔽,T,D,M,C,R,MT}(
        dims, base, conditioner, retraction, metric
    )
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
function get_natural_manifold(
    ::Type{T}, dims, conditioner=nothing, retraction=nothing
) where {T}
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

struct ChartNOrderRetraction{Order,E} <: AbstractRetractionMethod
    extra::E
end

function ChartNOrderRetraction{O}() where {O}
    return ChartNOrderRetraction{O,Nothing}(nothing)
end

const FirstOrderRetraction = ChartNOrderRetraction{1}
const SecondOrderRetraction = ChartNOrderRetraction{2}

"""
    SecondOrderRetraction(; backend=nothing)

Create a second-order retraction method that uses Christoffel symbols to compute
a more accurate retraction. If a backend is provided, it will be used for any
automatic differentiation needed to compute the Christoffel symbols.

# Arguments
- `backend`: Optional backend for automatic differentiation (e.g., `ADTypes.AutoForwardDiff()`)
"""
function SecondOrderRetraction(; backend=nothing)
    return ChartNOrderRetraction{2,typeof(backend)}(backend)
end

function ManifoldsBase.default_retraction_method(
    ::NaturalParametersManifold{𝔽,TD,D,M,C,Nothing,FisherInformationMetric}, ::Type{T}
) where {𝔽,T,TD,D,M,C}
    return FirstOrderRetraction()
end

function ManifoldsBase.default_retraction_method(
    M::NaturalParametersManifold{𝔽,TD,D,BM,C,R}, ::Type{T}
) where {𝔽,T,TD,D,BM,C,R}
    return getretraction(M)
end

function ManifoldsBase.retract_fused!(
    ::NaturalParametersManifold, q, p, X, t::Number, method::FirstOrderRetraction
)
    q .= p .+ t .* X
    return q
end

function ManifoldsBase.retract!(
    M::NaturalParametersManifold, q, p, X, method::FirstOrderRetraction
)
    return ManifoldsBase.retract_fused!(M, q, p, X, one(eltype(X)), method)
end

function ManifoldsBase.retract_fused!(
    M::NaturalParametersManifold{𝔽,T,D,BM,C,R,FisherInformationMetric}, 
    q, p, X, t::Number, 
    method::SecondOrderRetraction
) where {𝔽,T,D,BM,C,R}
    basis = ManifoldsBase.get_basis_default(M, p)
    Γ = Manifolds.christoffel_symbols_second(M, p, basis; backend=method.extra)
    
    Δ = similar(p)
    Manifolds.@einsum Δ[k] = -0.5 * Γ[k,i,j] * (t * X[i]) * (t * X[j])
    
    q .= p .+ t .* X .+ Δ
    return q
end

function ManifoldsBase.retract!(
    M::NaturalParametersManifold, q, p, X, method::SecondOrderRetraction
)
    return ManifoldsBase.retract_fused!(M, q, p, X, one(eltype(X)), method)
end

struct NaturalBasis{𝔽,VST<:VectorSpaceType} <: AbstractBasis{𝔽,VST}
    vector_space::VST
end

NaturalBasis(𝔽 = ℝ, vs::VectorSpaceType = TangentSpaceType()) = NaturalBasis{𝔽,typeof(vs)}(vs)
NaturalBasis{𝔽}(vs::VectorSpaceType = TangentSpaceType()) where {𝔽} = NaturalBasis{𝔽,typeof(vs)}(vs)

function ManifoldsBase.get_basis_default(
    M::NaturalParametersManifold{𝔽,T,D,MB,C,R,FisherInformationMetric}, p
) where {𝔽,T,D,MB,C,R}
    return NaturalBasis{𝔽}()
end

function Manifolds.local_metric(
    M::NaturalParametersManifold{𝔽,T,D,MB,C,R,FisherInformationMetric}, p, ::NaturalBasis
) where {𝔽,T,D,MB,C,R}
    ef = convert(ExponentialFamilyDistribution, M, p)
    return ExponentialFamily.fisherinformation(ef)
end

function Manifolds.inverse_local_metric(
    M::NaturalParametersManifold{𝔽,T,D,MB,C,R,FisherInformationMetric}, p, ::NaturalBasis
) where {𝔽,T,D,MB,C,R}
    ef = convert(ExponentialFamilyDistribution, M, p)
    return cholinv(ExponentialFamily.fisherinformation(ef))
end
