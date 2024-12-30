using Distributions, LinearAlgebra, Random, Plots
using ExponentialFamily, ExponentialFamilyManifolds
using Manopt, ForwardDiff, StableRNGs
using PGFPlotsX, RecursiveArrayTools

"""
Generic optimization function as shown in Listing 1 of the paper.
Performs gradient descent optimization on manifold M with objective f.
"""
function optimize(M, f, η_init; max_iters=500)
    g(M, η) = ForwardDiff.gradient(
        x -> f(M, x), η
    )
    
    stopping_criterion = StopAfterIteration(max_iters) | 
                        StopWhenGradientNormLess(1e-6)
    
    η_opt = gradient_descent(M, f, g, η_init;
                           stopping_criterion=stopping_criterion)
    
    return convert(Distribution, 
           convert(ExponentialFamilyDistribution, M, η_opt))
end

"""
MLE objective function as shown in Listing 2 of the paper.
"""
function f_mle(M, η, samples)
    dist = convert(ExponentialFamilyDistribution, M, η)
    dist = convert(Distribution, dist)
    return -mean(logpdf(dist, s) for s in eachcol(samples))
end

"""
VI objective function (Free Energy) as shown in Listing 3 of the paper.
"""
function f_vi(M, η, target_dist; n_samples=2000)
    q = convert(ExponentialFamilyDistribution, M, η)
    q_dist = convert(Distribution, q)
    
    # Sample from q using stable RNG for reproducibility
    samples = rand(StableRNG(22), q_dist, n_samples)
    
    # Compute difference of log probabilities
    diff(s) = logpdf(target_dist, s) - logpdf(q_dist, s)
    return -mean(diff(samples[:, i]) for i in 1:n_samples)
end

"""
Natural gradient implementation as shown in Listing 4 of the paper.
"""
function natural_gradient(M, η, f)
    q = convert(ExponentialFamilyDistribution, M, η)
    F = fisherinformation(q)
    grad = ForwardDiff.gradient(
        x -> f(M, x), η
    )
    return F \ grad
end

"""
Creates a mixture of two Gaussians as target distribution
for demonstrating the differences between MLE and VI.
"""
function create_target_distribution()
    μ1 = [0.9, 1.0]
    Σ1 = [1.0 0.2; 0.2 1.2]
    
    μ2 = [-1.3, -1]
    Σ2 = [1.2 0.3; 0.3 1.5]
    
    mixture = MixtureModel([
        MvNormal(μ1, Σ1),
        MvNormal(μ2, Σ2)
    ], [0.5, 0.5])
    
    return mixture
end

"""
Helper function to visualize and compare two different approximations
to the target distribution.
"""
function plot_comparison(q1, q2, target_dist, title_q1, title_q2)
    x = range(-6, 6, length=100)
    y = range(-6, 6, length=100)

    z_target = [pdf(target_dist, [xi, yi]) for yi in y, xi in x]
    z_p = [pdf(q1, [xi, yi]) for yi in y, xi in x]
    z_q = [pdf(q2, [xi, yi]) for yi in y, xi in x]

    plt1 = contour(x, y, z_target; fill=true, alpha=0.4, color=:viridis,
        levels=15, title=title_q1, xlabel="x1", ylabel="x2")
    contour!(plt1, x, y, z_p; color=:red, fill=false, linewidth=2, levels=10)

    plt2 = contour(x, y, z_target; fill=true, alpha=0.4, color=:viridis,
        levels=15, title=title_q2, xlabel="x1", ylabel="x2")
    contour!(plt2, x, y, z_q; color=:blue, fill=false, linewidth=2, levels=10)

    plot(plt1, plt2, layout=(1,2), size=(1200,400))
end

# Example usage reproducing paper figures
target_dist = create_target_distribution()
samples = rand(StableRNG(42), target_dist, 2000)

# Initialize parameters for full covariance Gaussian
form = MvNormalMeanCovariance
η_init = ArrayPartition([0.4, 0.4], [-0.5 0.0; 0.0 -0.5])

# MLE vs VI comparison (Figure 1 top)
M = ExponentialFamilyManifolds.get_natural_manifold(form, (2,))
q_mle = optimize(M, (M, η) -> f_mle(M, η, samples), η_init)
q_vi = optimize(M, (M, η) -> f_vi(M, η, target_dist), η_init)
plot_comparison(q_mle, q_vi, target_dist, "Maximum likelihood", "Free Energy")
savefig("comparison1.pdf")

# Isotropic vs Full covariance (Figure 1 bottom)
η_init_diag = ArrayPartition([0.4, 0.4], [-0.5])
M_diag = ExponentialFamilyManifolds.get_natural_manifold(MvNormalMeanScalePrecision, (2,))
q_isotropic = optimize(M_diag, (M, η) -> f_vi(M, η, target_dist), η_init_diag)
plot_comparison(q_isotropic, q_vi, target_dist, "Isotropic gaussian", "Full covariance")
savefig("comparison2.pdf")