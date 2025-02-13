```@meta
CurrentModule = ExponentialFamilyManifolds
```

# ExponentialFamilyManifolds

`ExponentialFamilyManifolds.jl` provides implementations of manifolds for the natural parameters of exponential family distributions, using `Manifolds.jl`. These manifolds are compatible with `ManifoldsBase.jl`, enabling optimization of the natural parameters of exponential family distributions using `Manopt.jl`.

The primary operation in the package is the [`get_natural_manifold`](@ref) function, which returns the appropriate manifold for the natural parameters of a given exponential family member type, its dimension and (if required) conditioner. 

```@docs 
ExponentialFamilyManifolds.get_natural_manifold
```

---

The [`get_natural_manifold`](@ref) function returns [`NaturalParametersManifold`](@ref) manifold, which is a wrapper around the actual manifold (the base) for the natural parameters that stores extra useful properties and provides the necessary operations for optimization with `Manopt.jl`. 

```@example api
using ExponentialFamilyManifolds, ExponentialFamily

ExponentialFamilyManifolds.get_natural_manifold(Beta, (), nothing)
```

```@docs
ExponentialFamilyManifolds.NaturalParametersManifold
```

Its not advised to use the [`NaturalParametersManifold`](@ref) to create a manifold, but instead use the [`get_natural_manifold`](@ref) function.

---

## Natural manifold base

The [`get_natural_manifold_base`](@ref) function returns the base manifold without the wrapper. 

```@example api
ExponentialFamilyManifolds.get_natural_manifold_base(Beta, (), nothing)
```

The base manifold, however, does not encode the information about the conditioner, hence, it cannot be used for all exponential members. Additionally, it does not encode the type of the underlying exponential family members. For instance, the `LogNormal` and the univariate `Normal` distribution share the same base manifold, yet they represent different members of the exponential family of distributions.

```@docs 
ExponentialFamilyManifolds.get_natural_manifold_base
```

---

## Product manifolds

Some base manifolds are known as _Product Manifolds_, which consist of several manifolds combined together. For example, the natural parameters of a multivariate Normal distribution form a product of a Euclidean vector manifold and a symmetric negative definite matrix manifold. The [`partition_point`](@ref) function takes a plain vector of natural parameters and (typically, but not always) returns a partitioned array for each submanifold in the form of an `ArrayPartition` from `RecursiveArrayTools.jl`.

```@example api
M = ExponentialFamilyManifolds.get_natural_manifold(Beta, (), nothing)
p = ExponentialFamilyManifolds.partition_point(M, [ 1.0, 2.0 ])
```

```@example api
typeof(p)
```

The partitioned point functions as a regular vector but encodes the structure of the product manifold, allowing differentiation between the submanifolds within the vector.

```@example api
ExponentialFamilyManifolds.ManifoldsBase.submanifold_component(p, 1)
```

```@example api
ExponentialFamilyManifolds.ManifoldsBase.submanifold_component(p, 2)
```

```@docs
ExponentialFamilyManifolds.partition_point
```

## Custom generic manifolds

`ExponentialFamilyManifolds.jl` introduces additional manifolds not included in `Manifolds.jl`. This is crucial because certain exponential family distributions have natural parameters that require specific manifolds, such as negative definite matrices for the multivariate Gaussian distribution. These manifolds do not implement every operation defined in `ManifoldsBase.jl`, but they do provide the essential operations needed for optimization with `Manopt.jl`.

```@docs
ExponentialFamilyManifolds.ShiftedPositiveNumbers
ExponentialFamilyManifolds.ShiftedNegativeNumbers
ExponentialFamilyManifolds.SymmetricNegativeDefinite
ExponentialFamilyManifolds.SinglePointManifold
ExponentialFamilyManifolds.NormalGammaNaturalManifold
```

## Optimization example

Suppose, we have a set of samples from a certain exponential family distribution and we want to estimate the natural parameters of the distribution using the `Manopt.jl` package.

```@example optimization
using ExponentialFamily, Distributions, Plots, StableRNGs

rng  = StableRNG(42)
dist = Beta(24, 6)
data = rand(rng, dist, 200)

histogram(data, xlim = (0, 1), label = "data", normalize=:pdf)
```

```@example optimization
using Manopt, ForwardDiff, ExponentialFamilyManifolds

# cost function
function f(M, p) 
    ef = convert(ExponentialFamilyDistribution, M, p)
    return -sum((d) -> logpdf(ef, d), data)
end

# gradient function
function g(M, p)
    return ForwardDiff.gradient((p) -> f(M, p), p)
end

M = ExponentialFamilyManifolds.get_natural_manifold(Beta, ())
p = rand(rng, M)
q = gradient_descent(M, f, g, p)

q_ef = convert(ExponentialFamilyDistribution, M, q)
q_η  = getnaturalparameters(q_ef)
```

Note that we performed the optimization in the natural parameters space, we can use `ExponentialFamily.jl` API to convert the vector fo natural parameters to the corresponding mean parameter space:

```@example optimization
map(NaturalParametersSpace() => MeanParametersSpace(), Beta, q_η)
```

As we can see the result is quite close to the actual distribution, which was used to generate the test data:

```@example optimization
params(MeanParametersSpace(), dist)
```

Let's also check the result, by plotting the estimated distribution on top of the data.

```@example optimization
histogram(data, xlim = (0, 1), label = "data", normalize=:pdf, fillalpha = 0.3)
plot!(0.0:0.01:1.0, (x) -> pdf(dist, x), label = "actual", fill = 0, fillalpha = 0.2)
plot!(0.0:0.01:1.0, (x) -> pdf(q_ef, x), label = "estimated", fill = 0, fillalpha = 0.5)
```

The difference in KL is quite small as well:

```@example optimization
using Test #hide
@test kldivergence(convert(Distribution, q_ef), dist) < 4e-3 #hide
kldivergence(convert(Distribution, q_ef), dist)
```

# Helpers 

```@docs 
ExponentialFamilyManifolds.Negated
```

# Index

```@index
```