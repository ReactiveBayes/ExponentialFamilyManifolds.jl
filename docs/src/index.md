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

## Advanced example

```@docs
forward_kl_objective
reverse_kl_objective
```

This example demonstrates the difference between Forward and Reverse KL divergence when fitting a Gaussian to a multimodal target distribution.

```@example kl_comparison
using Distributions, LinearAlgebra, Random, Plots
using ExponentialFamily, ExponentialFamilyManifolds
using Manopt, ForwardDiff, StableRNGs

# Create target distribution (mixture of 3 Gaussians in triangle formation)
function create_target_distribution()
    μ1 = [0.0, 0.0]    # First vertex
    μ2 = [2.0, 0.0]    # Second vertex
    μ3 = [1.0, 1.732]  # Third vertex (sqrt(3) ≈ 1.732)
    Σ = 0.2 * Matrix(I, 2, 2)
    w = fill(1/3, 3)   # Equal weights

    MixtureModel([
        MvNormal(μ1, Σ),
        MvNormal(μ2, Σ),
        MvNormal(μ3, Σ)
    ], w)
end

target_dist = create_target_distribution()
```

First, let's visualize our target distribution:

```@example kl_comparison
x = range(-6, 6, length=100)
y = range(-6, 6, length=100)
z_target = [pdf(target_dist, [xi, yi]) for yi in y, xi in x]

contour(x, y, z_target; 
    fill=true, color=:viridis,
    levels=15, title="Target Distribution",
    xlabel="x₁", ylabel="x₂"
)
```

Now we'll implement both Forward and Reverse KL objectives:

```@example kl_comparison
function forward_kl_objective(M, η, samples)
    q = convert(ExponentialFamilyDistribution, M, η)
    q_dist = convert(Distribution, q)
    -sum(logpdf(q_dist, s) for s in eachcol(samples))
end

function reverse_kl_objective(M, η, target_dist; n_samples=2000)
    q = convert(ExponentialFamilyDistribution, M, η)
    q_dist = convert(Distribution, q)
    
    samples = rand(StableRNG(422), q_dist, n_samples)
    log_ratios = [
        logpdf(target_dist, samples[:, i]) - logpdf(q_dist, samples[:, i])
        for i in 1:n_samples
    ]
    -mean(log_ratios)
end
```

Let's optimize using both approaches:

```@example kl_comparison
function optimize_distribution(form, target_dist, η_init; method=:forward, max_iters=100)
    M = ExponentialFamilyManifolds.get_natural_manifold(form, (2,))
    stopping_criterion = StopAfterIteration(max_iters) | StopWhenGradientNormLess(1e-6)

    if method == :forward
        samples = rand(StableRNG(42), target_dist, 2000)
        f(M, η) = forward_kl_objective(M, η, samples)
    else
        f(M, η) = reverse_kl_objective(M, η, target_dist)
    end
    
    g(M, η) = ForwardDiff.gradient(x -> f(M, x), η)
    
    η_opt = gradient_descent(M, f, g, η_init; stopping_criterion=stopping_criterion)
    convert(Distribution, convert(ExponentialFamilyDistribution, M, η_opt))
end

# Initialize and optimize
form = MvNormalMeanCovariance
η_init = ArrayPartition([0.4, 0.4], [-0.5 0.0; 0.0 -0.5])

q_forward = optimize_distribution(form, target_dist, η_init; method=:forward)
q_reverse = optimize_distribution(form, target_dist, η_init; method=:reverse)
```

Finally, let's compare the results:

```@example kl_comparison
function plot_fit(fitted_dist, target_dist, title)
    z_fitted = [pdf(fitted_dist, [xi, yi]) for yi in y, xi in x]
    
    contour(x, y, z_target; 
        fill=true, alpha=0.4, color=:viridis,
        levels=15, title=title, xlabel="x₁", ylabel="x₂"
    )
    contour!(x, y, z_fitted; 
        color=:red, fill=false, linewidth=2, levels=10
    )
end

p1 = plot_fit(q_forward, target_dist, "Forward KL")
p2 = plot_fit(q_reverse, target_dist, "Reverse KL")
plot(p1, p2, layout=(1,2), size=(1000,400))
```

The KL divergences show the quantitative difference:

```@example kl_comparison
println("Forward KL: ", round(kldivergence(q_forward, target_dist), digits=3))
println("Reverse KL: ", round(kldivergence(q_reverse, target_dist), digits=3))
```

# Helpers 

```@docs 
ExponentialFamilyManifolds.Negated
```

# Index

```@index
```