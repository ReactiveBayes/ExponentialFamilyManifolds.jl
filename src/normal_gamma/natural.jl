using ExponentialFamily
using FastCholesky
using LinearAlgebra

import ManifoldsBase: check_point, check_vector, manifold_dimension, representation_size, exp!, is_point, is_vector
import ManifoldsBase: zero_vector, zero_vector!, default_retraction_method
import ManifoldsBase: ExponentialRetraction
import ManifoldsBase: injectivity_radius

"""
Compute all Christoffel symbols Γ^k_{i j}(x).
Returns a 4×4×4 array: Gamma[k,i,j].
"""
function christoffel(::Type{NormalGamma}, ::NaturalParametersSpace, η)
    ef = ExponentialFamilyDistribution(NormalGamma, η, nothing)
    G = fisherinformation(ef)
    G⁻¹ = cholinv(G)
    dG  = partial_metric(NormalGamma, NaturalParametersSpace(), η) 
    # formula: Γ^k_{i,j} = 1/2 * sum_{ℓ} g^{kℓ} ( ∂i g_{ℓj} + ∂j g_{ℓi} - ∂ℓ g_{i j} )
    Γ = zeros(4,4,4)
    for k in 1:4
        for i in 1:4
            for j in 1:4
                s = 0.0
                for ℓ in 1:4
                    s += G⁻¹[k,ℓ] * (dG[ℓ,j,i] + dG[ℓ,i,j] - dG[i,j,ℓ])
                end
                Γ[k,i,j] = 0.5 * s
            end
        end
    end
    return Γ
end

function exp_secondorder(
    ::Type{NormalGamma},
    ::NaturalParametersSpace,
    p0,
    v0
 )
    Γ = christoffel(NormalGamma, NaturalParametersSpace(), p0)
    
    Δ = zeros(4)
    for k in 1:4
        for i in 1:4
            for j in 1:4
                Δ[k] += Γ[k,i,j] * v0[i] * v0[j]
            end
        end
        Δ[k] *= -0.5
    end
    
    return p0 + v0 + Δ
 end


"""
    NormalGammaNaturalManifold <: AbstractManifold{ℝ}

4D manifold for NormalGamma in natural params η = (η1, η2, η3, η4).
Domain constraints, etc. 
"""
struct NormalGammaNaturalManifold <: AbstractManifold{ℝ} end

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

function exp!(::NormalGammaNaturalManifold, η_out::AbstractVector, η_in::AbstractVector, X_in::AbstractVector, t::Real=1.0)
    η_out .= exp_secondorder(NormalGamma, NaturalParametersSpace(), η_in, t.*X_in)
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
    μ = rand(Euclidean())[1]
    λ = rand(PositiveNumbers())
    α = rand(PositiveNumbers())
    β = rand(PositiveNumbers())
    return collect(MeanToNatural(NormalGamma)((μ, λ, α, β)))
end

function Random.rand!(rng::AbstractRNG, M::NormalGammaNaturalManifold, η; kwargs...)
    μ = rand(rng, Euclidean())[1]
    λ = rand(rng, PositiveNumbers())
    α = rand(rng, PositiveNumbers()) 
    β = rand(rng, PositiveNumbers())

    η_nat = MeanToNatural(NormalGamma)((μ, λ, α, β))
    η .= η_nat
    return η
end

import ManifoldsBase: project!
 
function ManifoldsBase.project!(M::NormalGammaNaturalManifold, Y, p, X)
    Y .= X
    return Y
end