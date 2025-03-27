using Distributions, LinearAlgebra, Random
using ExponentialFamily, ExponentialFamilyManifolds
using Manopt, ForwardDiff, StableRNGs
using PGFPlotsX, RecursiveArrayTools
using ManifoldDiff, DifferentiationInterface
using ManifoldDiff: TangentDiffBackend
using FiniteDifferences

"""
Run optimization multiple times with different random initializations
and return the best result based on the objective function value.
"""
function optimize_multiple_starts(M, f, n_starts=10; rng=StableRNG(42))
    best_p = nothing
    best_value = Inf

    for _ in 1:n_starts
        p_init = rand(rng, M)
        p_opt = optimize(M, f, p_init)
        current_value = f(M, p_opt)

        if current_value < best_value
            best_value = current_value
            best_p = p_opt
        end
    end

    return convert(Distribution, convert(ExponentialFamilyDistribution, M, best_p))
end

"""
Generic optimization function as shown in Listing 1 of the paper.
Performs gradient descent optimization on manifold M with objective f.
"""
function optimize(M, f, p_init)
    autograd_backend = AutoFiniteDifferences(central_fdm(3, 1))
    backend = TangentDiffBackend(autograd_backend)
    grad = (M, p) -> ManifoldDiff.gradient(M, (p) -> f(M, p), p, backend)
    p_opt = gradient_descent(M, f, grad, p_init)
    return p_opt
end

"""
MLE objective function as shown in Listing 2 of the paper.
"""
function f_mle(M, η, samples)
    ef = convert(ExponentialFamilyDistribution, M, η)
    return -mean(logpdf(ef, s) for s in eachcol(samples))
end

"""
VI objective function (Free Energy) as shown in Listing 3 of the paper.
"""
function f_vi(M, η, target_dist; n_samples=2000)
    q = convert(ExponentialFamilyDistribution, M, η)
    q_dist = convert(Distribution, q)

    # Sample from q using stable RNG for reproducibility
    samples = rand(StableRNG(22), q_dist, n_samples)

    diff(s) = logpdf(q_dist, s) - logpdf(target_dist, s)
    return mean(diff(samples[:, i]) for i in 1:n_samples)
end

"""
Natural gradient implementation as shown in Listing 4 of the paper.
"""
function natural_gradient(M, η, f)
    q = convert(ExponentialFamilyDistribution, M, η)
    F = fisherinformation(q)
    grad = ForwardDiff.gradient(x -> f(M, x), η)
    return F \ grad
end

"""
Creates a mixture of two Gaussians as target distribution
for demonstrating the differences between MLE and VI.
"""
function create_target_distribution()
    μ1 = [3.5, 3.5]
    Σ1 = [1.0 0.2; 0.2 1.2]

    μ2 = [-1.3, -1]
    Σ2 = [1.2 0.3; 0.3 1.5]

    mixture = MixtureModel([MvNormal(μ1, Σ1), MvNormal(μ2, Σ2)], [0.4, 0.6])

    return mixture
end
# Example usage reproducing paper figures
target_dist = create_target_distribution()
samples = rand(StableRNG(42), target_dist, 2000);

# Initialize parameters for full covariance Gaussian
form = MvNormalMeanCovariance

# MLE vs VI comparison (Figure 1 top)
M = ExponentialFamilyManifolds.get_natural_manifold(form, (2,))

q_mle = optimize_multiple_starts(M, (M, p) -> f_mle(M, p, samples), 2);
q_vi = optimize_multiple_starts(M, (M, p) -> f_vi(M, p, target_dist), 1);

# Isotropic vs Full covariance (Figure 1 bottom)
p_init_diag = ArrayPartition([0.4, 0.4], [0.5])
M_diag = ExponentialFamilyManifolds.get_natural_manifold(MvNormalMeanScalePrecision, (2,))
p_isotropic = optimize(M_diag, (M, η) -> f_vi(M, η, target_dist), p_init_diag)
q_isotropic = convert(
    Distribution, convert(ExponentialFamilyDistribution, M_diag, p_isotropic)
)

include("plot_helper.jl")
plot_comparison(q_mle, q_vi, target_dist, "Maximum likelihood", "Free Energy")
savefig("comparison1.pdf")
plot_comparison(q_isotropic, q_vi, target_dist, "Isotropic gaussian", "Full covariance")
savefig("comparison2.pdf")
