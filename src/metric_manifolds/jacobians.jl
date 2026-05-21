"""
    jacobian_manifold_to_nat!(M, X_nat, X_manifold)

Transform tangent vector from manifold coordinates to natural parameter coordinates.
"""
function jacobian_manifold_to_nat(
    M::ExponentialFamilyManifolds.NaturalParametersManifold, X_manifold
)
    X_nat = jacobian_manifold_to_nat!(M, zero_vector(M, X_manifold), X_manifold)
    return X_nat
end

"""
    jacobian_nat_to_manifold(M, X_nat)

Transform tangent vector from natural parameter coordinates to manifold coordinates.
"""
function jacobian_nat_to_manifold(
    M::ExponentialFamilyManifolds.NaturalParametersManifold, X_nat
)
    X_manifold = jacobian_nat_to_manifold!(M, zero_vector(M, X_nat), X_nat)
    return X_manifold
end

"""
    jacobian_nat_to_manifold!(M, X_manifold, X_nat)

Transform tangent vector from natural parameter coordinates to default manifold coordinates.
For NormalMeanVariance: (dη₁, dη₂) → (dη₁, dλ) where dλ = -dη₂.
"""
function jacobian_nat_to_manifold!(
    ::ExponentialFamilyManifolds.NaturalParametersManifold{
        F,ExponentialFamily.NormalMeanVariance
    },
    X_manifold,
    X_nat,
) where {F}
    X_manifold[1:1] .= X_nat[1]
    X_manifold[2:2] .= -X_nat[2]
    return X_manifold
end

"""
    jacobian_manifold_to_nat!(M, X_nat, X_manifold)

Transform tangent vector from manifold coordinates to natural parameter coordinates.
For NormalMeanVariance: (dη₁, dλ) → (dη₁, dη₂) where dη₂ = -dλ.
"""
function jacobian_manifold_to_nat!(
    ::ExponentialFamilyManifolds.NaturalParametersManifold{
        F,ExponentialFamily.NormalMeanVariance
    },
    X_nat,
    X_manifold,
) where {F}
    X_nat[1:1] .= X_manifold[1]
    X_nat[2:2] .= -X_manifold[2]
    return X_nat
end

"""
    jacobian_nat_to_manifold!(M, X_manifold, X_nat)

Transform tangent vector from natural parameter coordinates to default manifold coordinates.
For NormalMeanVariance: (dη₁, dη₂) → (dη₁, dλ) where dλ = -dη₂.
"""
function jacobian_nat_to_manifold!(
    ::ExponentialFamilyManifolds.NaturalParametersManifold{
        F,ExponentialFamily.Bernoulli
    },
    X_manifold,
    X_nat,
) where {F}
    X_manifold .= X_nat[1]
    return X_manifold
end

"""
    jacobian_manifold_to_nat!(M, X_nat, X_manifold)

Transform tangent vector from manifold coordinates to natural parameter coordinates.
For NormalMeanVariance: (dη₁, dλ) → (dη₁, dη₂) where dη₂ = -dλ.
"""
function jacobian_manifold_to_nat!(
    ::ExponentialFamilyManifolds.NaturalParametersManifold{
        F,ExponentialFamily.Bernoulli
    },
    X_nat,
    X_manifold,
) where {F}
    X_nat .= X_manifold
    return X_nat
end