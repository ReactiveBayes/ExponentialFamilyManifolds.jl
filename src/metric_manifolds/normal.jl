using StaticArrays
using RecursiveArrayTools: ArrayPartition

# ============================================================================
# Helper functions: coordinate conversions and hyperbolic geometry
# ============================================================================

# ------------------------------------------------------------------
# Representation:
#   manifold point p = (η₁, λ), where λ = -η₂ > 0
#   natural parameters η = (η₁, η₂), η₂ < 0
# ------------------------------------------------------------------

@inline _point_to_eta(p) = SA[p[1], -p[2]]          # (η₁, λ) -> (η₁, η₂)
@inline _eta_to_point(η::SVector{2}) = SA[η[1], -η[2]]

@inline _X_to_eta(X) = SA[X[1], -X[2]]             # tangent: dη₁ = dη₁, dη₂ = -dλ
@inline _eta_to_X(Xη::SVector{2}) = SA[Xη[1], -Xη[2]]

# --- Basic conversions: natural ↔ mean/variance ↔ half-plane ---

# (η₁, η₂) -> (μ, σ²)
@inline function _eta_to_meanvariance(η1, η2)
    μ  = -η1 / (2 * η2)
    σ² = -inv(2 * η2)
    return μ, σ²
end

# (μ, σ²) -> (η₁, η₂)
@inline function _meanvariance_to_eta(μ, σ²)
    η1 = μ / σ²
    η2 = -inv(2 * σ²)
    return η1, η2
end

# (η₁, η₂) -> Poincaré half-plane coords (x,y), y>0
# here x = μ/√2, y = σ
@inline function _eta_to_halfplane(η::SVector{2,T}) where {T}
    η1, η2 = η
    μ, σ²  = _eta_to_meanvariance(η1, η2)
    @assert σ² > zero(T) "NormalMeanVariance natural parameters must satisfy η₂ < 0"
    σ      = sqrt(σ²)
    x      = μ / sqrt(T(2))
    y      = σ
    return SA[x, y]
end

# (x,y) in half-plane -> (η₁, η₂)
@inline function _halfplane_to_eta(xy::SVector{2,T}) where {T}
    x, y = xy
    μ    = sqrt(T(2)) * x
    σ²   = y^2
    η1, η2 = _meanvariance_to_eta(μ, σ²)
    return SA[η1, η2]
end

# --- Jacobians between natural params and half-plane coords ---

# J_η→(x,y) evaluated at η = (η₁, η₂)
#   [∂x/∂η₁  ∂x/∂η₂]
#   [∂y/∂η₁  ∂y/∂η₂]
@inline function _J_eta_to_xy(η::SVector{2})
    η1, η2 = η
    μ, σ²  = _eta_to_meanvariance(η1, η2)
    σ      = sqrt(σ²)
    c      = sqrt(2.0)
    a11 = -c / (4 * η2)
    a12 =  c * η1 / (4 * η2^2)
    a21 = 0.0
    a22 = -σ / (2 * η2)        # equals sqrt(2)/(4*(-η2)^(3/2))
    return SA[a11 a12;
              a21 a22]
end

# J_(x,y)→η evaluated at (x,y)
#   [∂η₁/∂x  ∂η₁/∂y]
#   [∂η₂/∂x  ∂η₂/∂y]
@inline function _J_xy_to_eta(xy::SVector{2})
    x, y = xy
    c  = sqrt(2.0)
    y2 = y^2
    y3 = y2 * y
    b11 =  c / y2
    b12 = -2c * x / y3
    b21 = 0.0
    b22 =  1.0 / y3
    return SA[b11 b12;
              b21 b22]
end

# --- Half-plane ↔ hyperboloid (curvature -1) ---

# (x,y) in upper half-plane -> hyperboloid point u ∈ ℝ³
# with Minkowski metric ⟨u,u⟩ = -u₀² + u₁² + u₂² = -1, u₀>0
@inline function _hp_to_hyp(xy::SVector{2})
    x, y = xy
    x2 = x^2
    y2 = y^2
    u0 = (x2 + y2 + 1) / (2*y)
    u1 = x / y
    u2 = (x2 + y2 - 1) / (2*y)
    return SA[u0, u1, u2]
end

# hyperboloid point -> (x,y) in half-plane
@inline function _hyp_to_hp(u::SVector{3})
    u0, u1, u2 = u
    denom = u0 - u2
    x = u1 / denom
    y = 1 / denom
    return SA[x, y]
end

# J_(x,y)→u (3×2)
@inline function _J_hp_to_hyp(xy::SVector{2})
    x, y = xy
    x2 = x^2
    y2 = y^2
    return SA[
        x/y                    (-x2 + y2 - 1) / (2*y2);
        1/y                    -x / y2;
        x/y                    (-x2 + y2 + 1) / (2*y2)
    ]
end

# J_u→(x,y) (2×3) = derivative of inverse map
@inline function _J_hyp_to_hp(u::SVector{3})
    u0, u1, u2 = u
    denom  = u0 - u2
    denom2 = denom^2
    return SA[
        -u1/denom2   1/denom   u1/denom2;
        -1/denom2    0.0       1/denom2
    ]
end

# --- Hyperbolic geometry on the hyperboloid ---

@inline _mdot(a::SVector{3}, b::SVector{3}) =
    -a[1]*b[1] + a[2]*b[2] + a[3]*b[3]  # Minkowski inner product

# Exponential on the hyperboloid (curvature -1)
function _hyperbolic_exp(u::SVector{3,T}, W::SVector{3,T}, t::T) where {T<:Real}
    m2 = _mdot(W,W)
    m2 = max(m2, zero(T))
    speed = sqrt(m2)
    if speed < eps(T)
        return u
    end
    ct = cosh(speed*t)
    st = sinh(speed*t)
    u_t = ct*u + (st/speed)*W
    return u_t
end

# Logarithmic map on the hyperboloid (Nagano et al., Eq. (6))
function _hyperbolic_log(u::SVector{3,T}, v::SVector{3,T}) where {T<:Real}
    if u == v
        return zero(u)
    end
    ip = _mdot(u,v)
    α  = -ip
    d  = acosh(α)
    w  = v - α*u          # v - α u, α = -⟨u,v⟩
    n2 = _mdot(w,w)
    if n2 <= zero(T)
        return zero(u)
    end
    n = sqrt(n2)
    return (d/n) * w
end

# Parallel transport on the hyperboloid (Nagano et al., Eq. (3))
function _hyperbolic_parallel_transport(
    ν::SVector{3,T},
    μ::SVector{3,T},
    v::SVector{3,T},
) where {T<:Real}
    α = -_mdot(ν, μ)
    num   = _mdot(μ - α*ν, v)
    coeff = num / (α + one(T))
    return v + coeff * (ν + μ)
end

# ============================================================================
# High-level geodesic and log map operations in natural coordinates
# ============================================================================

# Geodesic in η-coordinates (internal helper)
function geodesic_exact_eta(
    t::Real,
    η0::SVector{2,T},
    Xη0::SVector{2,T},
) where {T<:Real}
    # point: η -> (x,y) -> hyperboloid
    xy0 = _eta_to_halfplane(η0)
    u0  = _hp_to_hyp(xy0)

    # tangent: η -> (x,y) -> hyperboloid
    # Fisher metric = 2 × Poincaré, so scale by 1/√2 when pushing to hyperboloid
    Jη_xy = _J_eta_to_xy(η0)
    V0    = Jη_xy * Xη0
    Jxy_u = _J_hp_to_hyp(xy0)
    W0    = (Jxy_u * V0) / sqrt(T(2))

    # geodesic on hyperboloid
    u_t  = _hyperbolic_exp(u0, W0, T(t))

    # back to η
    xy_t = _hyp_to_hp(u_t)
    η_t  = _halfplane_to_eta(xy_t)

    return η_t
end

# Public geodesic: manifold coordinates p = (η₁, λ)
function geodesic_exact(
    t::Real,
    p,
    X,
)
    η0  = _point_to_eta(p)
    Xη0 = _X_to_eta(X)
    η_t = geodesic_exact_eta(t, η0, Xη0)
    p_t = _eta_to_point(η_t)
    return p_t
end

# Log map in η-coordinates
function log_map_eta(ηp::SVector{2,T}, ηq::SVector{2,T}) where {T<:Real}
    # points to half-plane and hyperboloid
    xyp = _eta_to_halfplane(ηp)
    xyq = _eta_to_halfplane(ηq)
    up  = _hp_to_hyp(xyp)
    uq  = _hp_to_hyp(xyq)

    # log on hyperboloid
    ξ   = _hyperbolic_log(up, uq)

    # back to (x,y) then η (tangent at p)
    # Fisher metric = 2 × Poincaré, so scale by √2 when pulling from hyperboloid
    Ju_xy = _J_hyp_to_hp(up)
    Vp    = Ju_xy * ξ
    Jxy_η = _J_xy_to_eta(xyp)
    Xη    = (Jxy_η * Vp) * sqrt(T(2))

    return Xη
end

# Log map in manifold coordinates p = (η₁, λ)
function log_map(p, q)
    ηp = _point_to_eta(p)
    ηq = _point_to_eta(q)
    Xη = log_map_eta(ηp, ηq)
    X  = _eta_to_X(Xη)
    return X
end

# ============================================================================
# ManifoldsBase method overrides
# ============================================================================

# Exponential map implementations

function ManifoldsBase.exp!(
    ::WithMetric{F,NormalMeanVariance,NaturalMetric},
    q,
    p,
    X,
) where {F}
    p_t = geodesic_exact(1, p, X)
    q[1] = p_t[1]
    q[2] = p_t[2]
    return q
end

function ManifoldsBase.exp(
    ::WithMetric{F,NormalMeanVariance,NaturalMetric},
    p,
    X,
) where {F}
    p_t = geodesic_exact(1, p, X)
    return ArrayPartition([p_t[1]], [p_t[2]])
end

function ManifoldsBase.exp_fused!(
    ::WithMetric{F,NormalMeanVariance,NaturalMetric},
    q,
    p,
    X,
    t::Number,
) where {F}
    p_t = geodesic_exact(t, p, X)
    q[1] = p_t[1]
    q[2] = p_t[2]
    return q
end

function ManifoldsBase.exp_fused(
    ::WithMetric{F,NormalMeanVariance,NaturalMetric},
    p,
    X,
    t::Number,
) where {F}
    p_t = geodesic_exact(t, p, X)
    return ArrayPartition([p_t[1]], [p_t[2]])
end

# Log map implementations

function ManifoldsBase.log!(
    ::WithMetric{F,NormalMeanVariance,NaturalMetric},
    X,
    p,
    q,
) where {F}
    X .= log_map(p, q)
    return X
end

function ManifoldsBase.log(
    ::WithMetric{F,NormalMeanVariance,NaturalMetric},
    p,
    q,
) where {F}
    X = log_map(p, q)
    return ArrayPartition([X[1]], [X[2]])
end

# Parallel transport in natural parameters

function ManifoldsBase.parallel_transport_to(
    M::WithMetric{F,NormalMeanVariance,NaturalMetric},
    p,
    X,
    q;
    kwargs...,
) where {F}
    ηp = _point_to_eta(p)
    ηq = _point_to_eta(q)
    Xη = _X_to_eta(X)

    # p, q to hyperboloid
    xyp = _eta_to_halfplane(ηp)
    xyq = _eta_to_halfplane(ηq)
    up  = _hp_to_hyp(xyp)
    uq  = _hp_to_hyp(xyq)

    # push X to hyperboloid (scale by 1/√2 for metric)
    Jη_xy = _J_eta_to_xy(ηp)
    Vp    = Jη_xy * Xη
    Jxy_u = _J_hp_to_hyp(xyp)
    Wp    = (Jxy_u * Vp) / sqrt(2.0)

    # parallel transport on hyperboloid
    Wq    = _hyperbolic_parallel_transport(up, uq, Wp)

    # pull back to η at q (scale by √2 for metric)
    Ju_xy = _J_hyp_to_hp(uq)
    Vq    = Ju_xy * Wq
    Jxy_η = _J_xy_to_eta(xyq)
    Xηq   = (Jxy_η * Vq) * sqrt(2.0)

    Xq    = _eta_to_X(Xηq)
    return ArrayPartition([Xq[1]], [Xq[2]])
end

function ManifoldsBase.parallel_transport_to!(
    M::WithMetric{F,NormalMeanVariance,NaturalMetric},
    Y,
    p,
    X,
    q;
    kwargs...,
) where {F}
    Y .= ManifoldsBase.parallel_transport_to(M, p, X, q; kwargs...)
    return Y
end

# Direction-based PT using exp

function ManifoldsBase.parallel_transport_direction(
    M::WithMetric{F,NormalMeanVariance,NaturalMetric},
    p,
    X,
    d;
    kwargs...,
) where {F}
    q = ManifoldsBase.exp(M, p, d)
    return ManifoldsBase.parallel_transport_to(M, p, X, q; kwargs...)
end

function ManifoldsBase.parallel_transport_direction!(
    M::WithMetric{F,NormalMeanVariance,NaturalMetric},
    Y,
    p,
    X,
    d;
    kwargs...,
) where {F}
    q = ManifoldsBase.exp(M, p, d)
    return ManifoldsBase.parallel_transport_to!(M, Y, p, X, q; kwargs...)
end

# vector_transport_to aliases ParallelTransport, like in your Bernoulli code

function ManifoldsBase.vector_transport_to(
    M::WithMetric{F,NormalMeanVariance,NaturalMetric},
    p,
    X,
    q,
    ::ManifoldsBase.ParallelTransport,
) where {F}
    return ManifoldsBase.parallel_transport_to(M, p, X, q)
end

function ManifoldsBase.vector_transport_to!(
    M::WithMetric{F,NormalMeanVariance,NaturalMetric},
    Y,
    p,
    X,
    q,
    ::ManifoldsBase.ParallelTransport,
) where {F}
    return ManifoldsBase.parallel_transport_to!(M, Y, p, X, q)
end