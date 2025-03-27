# Exponential Family Manifolds Example

This repository contains a practical implementation of optimization techniques on exponential family manifolds, reproducing the examples shown in our research paper. It demonstrates how to perform maximum likelihood estimation (MLE) and variational inference (VI) using natural gradient descent on manifolds of exponential family distributions.

## Overview

This example showcases:
- Optimization on manifolds of exponential family distributions
- Maximum likelihood estimation for density estimation
- Variational inference with Free Energy objective
- Natural gradient optimization
- Comparison between isotropic and full covariance Gaussian approximations

## Usage

First, navigate to the `paper_example` directory and activate the local environment:

```bash
cd path/to/ExponentialFamilyManifolds.jl/paper_example
```

Once in the Julia REPL with the project activated, you can install or update dependencies:

```julia
using Pkg
Pkg.instantiate()  # This will install all dependencies as specified in Project.toml
```

And run the example script from within the Julia REPL once you activated the project:

```julia
include("paper_example.jl")
```

This will:
1. Create a mixture of two Gaussians as the target distribution
2. Perform MLE and VI to approximate this distribution
3. Compare isotropic and full covariance Gaussian models
4. Generate visualization plots saved as PDF files

## Example Explanation

The script implements several key functions:

- `optimize_multiple_starts`: Performs optimization with multiple random initializations
- `optimize`: Generic gradient-based optimization on manifolds
- `f_mle`: Maximum likelihood objective function
- `f_vi`: Variational inference objective function (Free Energy)

We demonstrate these techniques on a 2D mixture of Gaussians, showing how different approaches approximate the target distribution.

## Output

The example produces two comparison plots:
1. `comparison1.pdf`: Compares maximum likelihood estimation vs. Free Energy minimization
2. `comparison2.pdf`: Compares isotropic Gaussian vs. full covariance approximations

The contour plots show how well each method captures the underlying mixture distribution.

## Performance Notes

The current implementation prioritizes clarity over performance. For larger problems, consider:
- Parallelizing the multiple starts
- Using precomputed values for repeated operations
- Replacing generator expressions with pre-allocated arrays
- Employing more sophisticated optimization algorithms