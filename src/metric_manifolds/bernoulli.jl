@inline ManifoldsBase.get_forwarding_type(::WithMetric{F, Bernoulli, NaturalMetric}, ::Any, ::typeof(exp)) where {F} = _STOP
@inline ManifoldsBase.get_forwarding_type(::WithMetric{F, Bernoulli, NaturalMetric}, ::Any, ::typeof(Manifolds.exp_fused)) where {F} = _STOP
@inline ManifoldsBase.get_forwarding_type(::WithMetric{F, Bernoulli, NaturalMetric}, ::Any, ::typeof(Manifolds.geodesic)) where {F} = _STOP
@inline ManifoldsBase.get_forwarding_type(::WithMetric{F, Bernoulli, NaturalMetric}, ::Any, ::typeof(log)) where {F} = _STOP
@inline ManifoldsBase.get_forwarding_type(::WithMetric{F, Bernoulli, NaturalMetric}, ::typeof(exp)) where {F} = _STOP
@inline ManifoldsBase.get_forwarding_type(::WithMetric{F, Bernoulli, NaturalMetric}, ::typeof(Manifolds.exp_fused)) where {F} = _STOP
@inline ManifoldsBase.get_forwarding_type(::WithMetric{F, Bernoulli, NaturalMetric}, ::typeof(Manifolds.geodesic)) where {F} = _STOP
@inline ManifoldsBase.get_forwarding_type(::WithMetric{F, Bernoulli, NaturalMetric}, ::typeof(log))  where {F} = _STOP


function _geodesic_eta(eta0, X)
    t = 1
    sqrtg0 = exp.(eta0/2) / (1 .+ exp.(eta0))
    s0     = 2.0.*atan.(exp.(eta0/2))
    s      = s0 .+ t .* sqrtg0 .* X
    2*log.(tan.(s./2))
end

# Log map (t=1): solve for X given eta1
function _log_eta(eta0, eta1)
    sqrtg0 = exp(eta0/2) / (1 + exp(eta0))
    s0     = 2*atan(exp(eta0/2))
    s1     = 2*atan(exp(eta1/2))
    (s1 - s0) / sqrtg0
end

ManifoldsBase.exp!(::WithMetric{F, Bernoulli, NaturalMetric}, p, X) where {F} = begin
    error("I call error here!")
    _geodesic_eta(p, X)
end

ManifoldsBase.exp(::WithMetric{F, Bernoulli, NaturalMetric}, p, X) where {F} = begin
    error("I call error here!")
    _geodesic_eta(p, X)
end

ManifoldsBase.log(::WithMetric{F, Bernoulli, NaturalMetric}, p, q) where {F} = begin
    _log_eta(p, q)
end
