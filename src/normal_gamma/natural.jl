using ExponentialFamily
using FastCholesky
using Distributions
using LinearAlgebra

import ManifoldsBase: check_point, check_vector, manifold_dimension, representation_size, exp!, is_point, is_vector
import ManifoldsBase: zero_vector, zero_vector!, default_retraction_method
import ManifoldsBase: ExponentialRetraction
import ManifoldsBase: injectivity_radius
import ManifoldsBase: AbstractBasis
import ManifoldsBase: VectorSpaceType, TangentSpaceType
import Manifolds: christoffel_symbols_second

function exp_secondorder(
    ::Type{NormalGamma},
    ::NaturalParametersSpace,
    Γ,
    p0,
    v0
)   
    Δ = similar(p0)  # Preallocate Δ with same type/size as p0
    Manifolds.@einsum Δ[k] = -0.5 * Γ[k,i,j] * v0[i] * v0[j]
    return p0 + v0 + Δ
end

struct ScoreBasis{ℝ,VST<:VectorSpaceType} <: AbstractBasis{ℝ,VST}
    vector_space::VST
end

function ScoreBasis(vst::VST) where {VST<:VectorSpaceType}
    return ScoreBasis{ℝ,VST}(vst)
end

"""
    NormalGammaNaturalManifold <: AbstractManifold{ℝ}

4D manifold for NormalGamma in natural params η = (η1, η2, η3, η4) with fisher information metric.

(see ExponentialFamily.jl/src/normal_gamma.jl)
"""
struct NormalGammaNaturalManifold{ℝ, T, B} <: AbstractManifold{ℝ}
    ensure_positivity_shift::T
    basis::B
end

function NormalGammaNaturalManifold()
    return NormalGammaNaturalManifold{ℝ, Float64, ScoreBasis{ℝ, TangentSpaceType}}(0.1, ScoreBasis(TangentSpaceType()))
end

function Manifolds.local_metric(
    ::NormalGammaNaturalManifold,
    p,
    ::ScoreBasis{ℝ, TangentSpaceType}
)
    ef = ExponentialFamilyDistribution(NormalGamma, p, nothing)
    return fisherinformation(ef)
end

function Manifolds.local_metric_jacobian(
    M::NormalGammaNaturalManifold,
    p,
    ::ScoreBasis{ℝ, TangentSpaceType};
    kwargs...
)
    return partial_metric(NormalGamma, NaturalParametersSpace(), p)
end

function manifold_dimension(::NormalGammaNaturalManifold)
    return 4
end

function representation_size(::NormalGammaNaturalManifold)
    return (4,)
end

function check_point(::NormalGammaNaturalManifold, η)
    cond_isproper = isproper(NaturalParametersSpace(), NormalGamma, η, nothing)
    if !cond_isproper
        return DomainError("$η is not a valid point on NormalGammaNaturalManifold")
    end
    return nothing
end

function check_vector(::NormalGammaNaturalManifold, η, X)
    if length(X)!=4
        return DomainError(X, "Vector must be length 4.")
    end
    return nothing
end

function exp!(M::NormalGammaNaturalManifold, η_out::AbstractVector, η_in::AbstractVector, X_in::AbstractVector, t::Real=1.0)
    Γ = christoffel_symbols_second(M, η_in, M.basis)
    η_out .= exp_secondorder(NormalGamma, NaturalParametersSpace(), Γ, η_in, t.*X_in)
    return η_out
end

# Provide a zero_vector etc.
function zero_vector(::NormalGammaNaturalManifold, η)
    return zeros(4)
end

function zero_vector!(::NormalGammaNaturalManifold, X, η)
    fill!(X, 0.0)
    return X
end

default_retraction_method(::NormalGammaNaturalManifold) = ExponentialRetraction()

injectivity_radius(::NormalGammaNaturalManifold) = Inf

import ManifoldsBase: inner

function inner(::NormalGammaNaturalManifold, p, X, Y)
    ef = ExponentialFamilyDistribution(NormalGamma, p, nothing)
    G = fisherinformation(ef)
    return dot(X, G, Y)
end

function Random.rand(M::NormalGammaNaturalManifold; kwargs...)
    return rand(Random.default_rng(), M; kwargs...)
end

function Random.rand(rng::AbstractRNG, M::NormalGammaNaturalManifold; kwargs...)
    # e.g. draw (μ,λ,α,β) from some easy distributions:
    μ = rand(rng, Normal(0,1))
    λ = rand(rng, Exponential(1)) + M.ensure_positivity_shift
    α = rand(rng, Exponential(1)) + M.ensure_positivity_shift
    β = rand(rng, Exponential(1)) + M.ensure_positivity_shift
    # Then transform to natural coords:
    η = MeanToNatural(NormalGamma)((μ, λ, α, β))
    return collect(η)
end

function Random.rand!(rng::AbstractRNG, M::NormalGammaNaturalManifold, η; kwargs...)
    μ = rand(rng, Normal(0,1))
    λ = rand(rng, Exponential(1)) + M.ensure_positivity_shift
    α = rand(rng, Exponential(1)) + M.ensure_positivity_shift
    β = rand(rng, Exponential(1)) + M.ensure_positivity_shift
    η_nat = MeanToNatural(NormalGamma)((μ, λ, α, β))
    η .= η_nat
    return η
end

import ManifoldsBase: project!
 
function ManifoldsBase.project!(::NormalGammaNaturalManifold, Y, p, X)
    Y .= X
    return Y
end