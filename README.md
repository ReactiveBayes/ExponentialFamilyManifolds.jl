# ExponentialFamilyManifolds

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ReactiveBayes.github.io/ExponentialFamilyManifolds.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ReactiveBayes.github.io/ExponentialFamilyManifolds.jl/dev/)
[![Build Status](https://github.com/ReactiveBayes/ExponentialFamilyManifolds.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ReactiveBayes/ExponentialFamilyManifolds.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ReactiveBayes/ExponentialFamilyManifolds.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ReactiveBayes/ExponentialFamilyManifolds.jl)
[![PkgEval](https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/E/ExponentialFamilyManifolds.svg)](https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/E/ExponentialFamilyManifolds.html)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

`ExponentialFamilyManifolds.jl` provides implementations of manifolds for the natural parameters of exponential family distributions, using `Manifolds.jl`. These manifolds are compatible with `ManifoldsBase.jl`, enabling optimization of the natural parameters of exponential family distributions using `Manopt.jl`.

Refer to the documentation for more information and optimization examples.

## Getting Started

First, install the package:

```julia
using Pkg
Pkg.add("ExponentialFamilyManifolds")
```

To use the package, you'll need to have the distribution types exported from `ExponentialFamily.jl` and methods from `Manifolds.jl`. Here's a basic example showing how to work with a Beta distribution:

```julia
using ExponentialFamily, ExponentialFamilyManifolds, Manifolds

# Create a Beta distribution
dist = Beta(2.0, 3.0)

# Convert to exponential family form
ef = convert(ExponentialFamilyDistribution, dist)

# Get the natural manifold for the Beta distribution
M = ExponentialFamilyManifolds.get_natural_manifold(Beta, ())

# Get natural parameters
η = getnaturalparameters(ef)

# Create a point on the manifold
p = ExponentialFamilyManifolds.partition_point(M, η)

# Create a tangent vector at point p
X = rand(M, vector_at = p)

# Move along the manifold in direction X with step size 0.1
q = Manifolds.retract_fused(M, p, X, 0.1)

# Convert back to exponential family distribution
ef_new = convert(ExponentialFamilyDistribution, M, q)
```

For a more advanced example, you can use `Manopt.jl` to optimize the natural parameters of exponential family distributions. Here's what you can do:

- Optimize maximum likelihood estimation
- Work with different cost functions on the manifold
- Use various optimization algorithms (gradient descent, conjugate gradient, etc.)

Check out our [optimization example](https://reactivebayes.github.io/ExponentialFamilyManifolds.jl/dev/#Optimization-example) in the documentation to learn more.

## Contributing and Support

We welcome contributions and questions from the community! Here's how you can get involved:

### Getting Help and Support

- **Q&A and Discussions**: Have questions or want to discuss ideas? Join our [discussions](https://github.com/ReactiveBayes/ExponentialFamilyManifolds.jl/discussions/categories/announcements) where you can:
  - Ask questions in the Q&A section
  - Share ideas and proposals
  - Show your work with the package
  - Stay updated with announcements

- **Issues**: Found a bug or want to request a feature? Please [open an issue](https://github.com/ReactiveBayes/ExponentialFamilyManifolds.jl/issues) on our GitHub repository.

### Contributing

Contributions are encouraged and appreciated! Here's how you can contribute:

1. Fork the repository and create your feature branch from `main`
2. Write clear, documented code and add tests for new functionality
3. Ensure all tests pass
4. Submit a pull request with a clear description of your changes

For more detailed information about contributing, please check our [discussions](https://github.com/ReactiveBayes/ExponentialFamilyManifolds.jl/discussions) page.

## Links

- Announcements: [https://github.com/ReactiveBayes/ExponentialFamilyManifolds.jl/discussions/categories/announcements](https://github.com/ReactiveBayes/ExponentialFamilyManifolds.jl/discussions/categories/announcements)
- Ideas: [https://github.com/ReactiveBayes/ExponentialFamilyManifolds.jl/discussions/categories/announcements](https://github.com/ReactiveBayes/ExponentialFamilyManifolds.jl/discussions/categories/announcements)
- Q&A: [https://github.com/ReactiveBayes/ExponentialFamilyManifolds.jl/discussions/categories/announcements](https://github.com/ReactiveBayes/ExponentialFamilyManifolds.jl/discussions/categories/announcements)
- Show and Tell: [https://github.com/ReactiveBayes/ExponentialFamilyManifolds.jl/discussions/categories/announcements](https://github.com/ReactiveBayes/ExponentialFamilyManifolds.jl/discussions/categories/announcements)
- Issues: [https://github.com/ReactiveBayes/ExponentialFamilyManifolds.jl/issues](https://github.com/ReactiveBayes/ExponentialFamilyManifolds.jl/issues)