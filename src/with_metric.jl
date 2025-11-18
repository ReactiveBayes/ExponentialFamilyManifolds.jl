const _STOP = ManifoldsBase.StopForwardingType()
const _FWD = ManifoldsBase.SimpleForwardingType()

abstract type MetricMode end
struct NaturalMetric <: MetricMode end  # Fisher info in active coordinates

struct WithMetric{𝔽,T,Mode<:MetricMode,M<:NaturalParametersManifold} <:
       AbstractDecoratorManifold{𝔽}
    man::M
end
ManifoldsBase.decorated_manifold(W::WithMetric) = W.man

function get_fisher_manifold(::Type{T}, dims, conditioner=nothing) where {T}
    natural_manifold = ExponentialFamilyManifolds.get_natural_manifold(T, dims, conditioner)
    fisher_manifold = with_natural_metric(natural_manifold)
    return fisher_manifold
end

function with_natural_metric(M::NaturalParametersManifold{F}) where {F}
    ef_typetag = ExponentialFamily.exponential_family_typetag(M)
    WithMetric{F,ef_typetag,NaturalMetric,typeof(M)}(M)
end

# When NaturalMetric is active, do NOT forward inner/norm. We implement them on the wrapper.
@inline ManifoldsBase.get_forwarding_type(
    ::WithMetric{F,T,NaturalMetric}, ::Any, ::typeof(inner)
) where {F,T} = _STOP
@inline ManifoldsBase.get_forwarding_type(
    ::WithMetric{F,T,NaturalMetric}, ::Any, ::typeof(norm)
) where {F,T} = _STOP
@inline ManifoldsBase.get_forwarding_type(
    ::WithMetric{F,T,NaturalMetric}, ::typeof(inner)
) where {F,T} = _STOP
@inline ManifoldsBase.get_forwarding_type(
    ::WithMetric{F,T,NaturalMetric}, ::typeof(norm)
) where {F,T} = _STOP

# Everything else still forwards.
@inline ManifoldsBase.get_forwarding_type(::WithMetric, ::Any) = _FWD
@inline ManifoldsBase.get_forwarding_type(::WithMetric, ::Any, ::Type) = _FWD

ExponentialFamily.exponential_family_typetag(::WithMetric{F,T}) where {F,T} = T

function Base.convert(::Type{ExponentialFamilyDistribution}, M::WithMetric, p)
    natural_manifold = M.man
    return ExponentialFamilyDistribution(
        exponential_family_typetag(natural_manifold),
        transform_back(natural_manifold, p),
        getconditioner(natural_manifold),
        nothing,
    )
end

function ManifoldsBase.norm(M::WithMetric{F,T,NaturalMetric}, p, X) where {F,T}
    return sqrt(inner(M, p, X, X))
end

# Default implementation for other distributions
function ManifoldsBase.inner(M::WithMetric{F,T,NaturalMetric}, p, X, Y) where {F,T}
    ef = convert(ExponentialFamilyDistribution, M, p)
    fisher_info = fisherinformation(ef)
    return dot(X, fisher_info, Y)
end
