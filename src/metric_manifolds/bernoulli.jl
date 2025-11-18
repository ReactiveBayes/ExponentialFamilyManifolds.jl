function _geodesic_exact_bernoulli(t, x0, v0)
    # Compute metric at initial point
    g0 = exp(x0) / (1 + exp(x0))^2

    # Energy constant
    E = g0 * v0^2

    # Integration constant from initial condition
    C = 2 * atan(exp(x0/2))

    # Sign based on initial velocity
    sign_v = sign(v0)

    # Position
    x = 2 * log(abs(tan((sign_v * sqrt(E) * t + C) / 2)))

    # Velocity from energy conservation
    g_x = exp(x) / (1 + exp(x))^2
    v = sign_v * sqrt(E / g_x)

    return x, v
end

function _log_map_bernoulli(x0, x1)
    C = 2 * atan(exp(x0/2))
    C1 = 2 * atan(exp(x1/2))

    # Determine sign (direction)
    sign_v = sign(C1 - C)

    # Solve for sqrt(E)
    sqrt_E = abs(C1 - C)

    # Compute v0 from E = g(x0) * v0^2
    g0 = exp(x0) / (1 + exp(x0))^2
    v0 = sign_v * sqrt(sqrt_E^2 / g0)

    return v0
end

# Exponential map implementations - these override the default forwarding behavior
function ManifoldsBase.exp!(::WithMetric{F,Bernoulli,NaturalMetric}, q, p, X) where {F}
    x, _ = _geodesic_exact_bernoulli(1, p[1], X[1])
    q .= x
    return q
end

function ManifoldsBase.exp(::WithMetric{F,Bernoulli,NaturalMetric}, p, X) where {F}
    x, _ = _geodesic_exact_bernoulli(1, p[1], X[1])
    return [x]
end

function ManifoldsBase.exp_fused!(
    ::WithMetric{F,Bernoulli,NaturalMetric}, q, p, X, t::Number
) where {F}
    x, _ = _geodesic_exact_bernoulli(1, p[1], X[1])
    q .= x
    return q
end

function ManifoldsBase.exp_fused(
    ::WithMetric{F,Bernoulli,NaturalMetric}, p, X, t::Number
) where {F}
    x, _ = _geodesic_exact_bernoulli(t, p[1], X[1])
    return [x]
end

function ManifoldsBase.log!(::WithMetric{F,Bernoulli,NaturalMetric}, X, p, q) where {F}
    X .= _log_map_bernoulli(p[1], q[1])
    return X
end

function ManifoldsBase.log(::WithMetric{F,Bernoulli,NaturalMetric}, p, q) where {F}
    return _log_map_bernoulli(p[1], q[1])
end

_sqrtg_eta(η) = 0.5 .* sech.(η ./ 2)

function ManifoldsBase.parallel_transport_to(
    ::WithMetric{F,Bernoulli,NaturalMetric}, p, X, q; kwargs...
) where {F}
    s_p = _sqrtg_eta(p)
    s_q = _sqrtg_eta(q)
    X .* (s_p ./ s_q)
end

function ManifoldsBase.parallel_transport_to!(
    ::WithMetric{F,Bernoulli,NaturalMetric}, Y, p, X, q; kwargs...
) where {F}
    # error("Call here")
    s_p = _sqrtg_eta(p)
    s_q = _sqrtg_eta(q)
    @. Y = X * (s_p / s_q)
    Y
end

# Parallel transport in a direction
function ManifoldsBase.parallel_transport_direction(
    M::WithMetric{F,Bernoulli,NaturalMetric}, p, X, d; kwargs...
) where {F}
    q = exp(M, p, d)
    return parallel_transport_to(M, p, X, q)
end

function ManifoldsBase.parallel_transport_direction!(
    M::WithMetric{F,Bernoulli,NaturalMetric}, Y, p, X, d; kwargs...
) where {F}
    q = exp(M, p, d)
    return parallel_transport_to!(M, Y, p, X, q)
end

# Vector transport with ParallelTransport method - this is what check_geodesic uses
function ManifoldsBase.vector_transport_to(
    M::WithMetric{F,Bernoulli,NaturalMetric}, p, X, q, ::ManifoldsBase.ParallelTransport
) where {F}
    return parallel_transport_to(M, p, X, q)
end

function ManifoldsBase.vector_transport_to!(
    M::WithMetric{F,Bernoulli,NaturalMetric}, Y, p, X, q, ::ManifoldsBase.ParallelTransport
) where {F}
    return parallel_transport_to!(M, Y, p, X, q)
end
