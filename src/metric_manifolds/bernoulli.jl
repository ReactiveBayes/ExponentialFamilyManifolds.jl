using ManifoldsBase

# Stop forwarding for functions we implement
@inline ManifoldsBase.get_forwarding_type(
    ::WithMetric{F,Bernoulli,NaturalMetric}, ::typeof(ManifoldsBase.exp), ::Type
) where {F} = _STOP
@inline ManifoldsBase.get_forwarding_type(
    ::WithMetric{F,Bernoulli,NaturalMetric}, ::typeof(ManifoldsBase.exp!), ::Type
) where {F} = _STOP
@inline ManifoldsBase.get_forwarding_type(
    ::WithMetric{F,Bernoulli,NaturalMetric}, ::typeof(ManifoldsBase.exp_fused), ::Type
) where {F} = _STOP
@inline ManifoldsBase.get_forwarding_type(
    ::WithMetric{F,Bernoulli,NaturalMetric}, ::typeof(ManifoldsBase.exp_fused!), ::Type
) where {F} = _STOP
@inline ManifoldsBase.get_forwarding_type(
    ::WithMetric{F,Bernoulli,NaturalMetric}, ::typeof(ManifoldsBase.log), ::Type
) where {F} = _STOP
@inline ManifoldsBase.get_forwarding_type(
    ::WithMetric{F,Bernoulli,NaturalMetric}, ::typeof(ManifoldsBase.log!), ::Type
) where {F} = _STOP
@inline ManifoldsBase.get_forwarding_type(
    ::WithMetric{F,Bernoulli,NaturalMetric},
    ::typeof(ManifoldsBase.parallel_transport_to),
    ::Type,
) where {F} = _STOP
@inline ManifoldsBase.get_forwarding_type(
    ::WithMetric{F,Bernoulli,NaturalMetric},
    ::typeof(ManifoldsBase.parallel_transport_to!),
    ::Type,
) where {F} = _STOP
@inline ManifoldsBase.get_forwarding_type(
    ::WithMetric{F,Bernoulli,NaturalMetric},
    ::typeof(ManifoldsBase.parallel_transport_direction),
    ::Type,
) where {F} = _STOP
@inline ManifoldsBase.get_forwarding_type(
    ::WithMetric{F,Bernoulli,NaturalMetric},
    ::typeof(ManifoldsBase.parallel_transport_direction!),
    ::Type,
) where {F} = _STOP
@inline ManifoldsBase.get_forwarding_type(
    ::WithMetric{F,Bernoulli,NaturalMetric},
    ::typeof(ManifoldsBase.vector_transport_to),
    ::Type,
) where {F} = _STOP
@inline ManifoldsBase.get_forwarding_type(
    ::WithMetric{F,Bernoulli,NaturalMetric},
    ::typeof(ManifoldsBase.vector_transport_to!),
    ::Type,
) where {F} = _STOP

# Also catch the single-argument versions (without Type)
@inline ManifoldsBase.get_forwarding_type(
    ::WithMetric{F,Bernoulli,NaturalMetric}, ::typeof(ManifoldsBase.exp)
) where {F} = _STOP
@inline ManifoldsBase.get_forwarding_type(
    ::WithMetric{F,Bernoulli,NaturalMetric}, ::typeof(ManifoldsBase.exp!)
) where {F} = _STOP
@inline ManifoldsBase.get_forwarding_type(
    ::WithMetric{F,Bernoulli,NaturalMetric}, ::typeof(ManifoldsBase.exp_fused)
) where {F} = _STOP
@inline ManifoldsBase.get_forwarding_type(
    ::WithMetric{F,Bernoulli,NaturalMetric}, ::typeof(ManifoldsBase.exp_fused!)
) where {F} = _STOP
@inline ManifoldsBase.get_forwarding_type(
    ::WithMetric{F,Bernoulli,NaturalMetric}, ::typeof(ManifoldsBase.log)
) where {F} = _STOP
@inline ManifoldsBase.get_forwarding_type(
    ::WithMetric{F,Bernoulli,NaturalMetric}, ::typeof(ManifoldsBase.log!)
) where {F} = _STOP
@inline ManifoldsBase.get_forwarding_type(
    ::WithMetric{F,Bernoulli,NaturalMetric}, ::typeof(ManifoldsBase.parallel_transport_to)
) where {F} = _STOP
@inline ManifoldsBase.get_forwarding_type(
    ::WithMetric{F,Bernoulli,NaturalMetric}, ::typeof(ManifoldsBase.parallel_transport_to!)
) where {F} = _STOP
@inline ManifoldsBase.get_forwarding_type(
    ::WithMetric{F,Bernoulli,NaturalMetric},
    ::typeof(ManifoldsBase.parallel_transport_direction),
) where {F} = _STOP
@inline ManifoldsBase.get_forwarding_type(
    ::WithMetric{F,Bernoulli,NaturalMetric},
    ::typeof(ManifoldsBase.parallel_transport_direction!),
) where {F} = _STOP
@inline ManifoldsBase.get_forwarding_type(
    ::WithMetric{F,Bernoulli,NaturalMetric}, ::typeof(ManifoldsBase.vector_transport_to)
) where {F} = _STOP
@inline ManifoldsBase.get_forwarding_type(
    ::WithMetric{F,Bernoulli,NaturalMetric}, ::typeof(ManifoldsBase.vector_transport_to!)
) where {F} = _STOP

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

function ManifoldsBase.exp_fused!(
    ::WithMetric{F,Bernoulli,NaturalMetric}, q, p, X, t::Number
) where {F}
    x, _ = _geodesic_exact_bernoulli(t, p[1], X[1])
    q .= x
    return q
end

function ManifoldsBase.log!(::WithMetric{F,Bernoulli,NaturalMetric}, X, p, q) where {F}
    X .= _log_map_bernoulli(p[1], q[1])
    return X
end

_sqrtg_eta(η) = 0.5 .* sech.(η ./ 2)

function ManifoldsBase.parallel_transport_to(
    ::WithMetric{F,Bernoulli,NaturalMetric}, p, X, q; kwargs...
) where {F}
    s_p = _sqrtg_eta(p)
    s_q = _sqrtg_eta(q) 
    return X .* (s_p ./ s_q)
end

function ManifoldsBase.parallel_transport_to!(
    M::WithMetric{F,Bernoulli,NaturalMetric}, Y, p, X, q; kwargs...
) where {F}
    Y .= ManifoldsBase.parallel_transport_to(M, p, X, q; kwargs...)
    return Y
end
