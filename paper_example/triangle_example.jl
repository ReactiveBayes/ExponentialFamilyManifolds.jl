using Distributions, LinearAlgebra, Random, Plots, RecursiveArrayTools
using ExponentialFamily, ExponentialFamilyManifolds
using Manopt, ForwardDiff, StableRNGs
using PGFPlotsX

"""
    Two intersecting Gaussians blobs with different means and covariances.
"""
function create_target_distribution()
    μ1 = [0.9, 1.0]
    Σ1 = [1.0 0.2; 0.2 1.2]
    
    μ2 = [-1.3, -1]
    Σ2 = [1.2 0.3; 0.3 1.5]
    
    mixture = MixtureModel([
        MvNormal(μ1, Σ1),
        MvNormal(μ2, Σ2)
    ], [0.5, 0.5])  # weights sum to 1
    
    return mixture
end

"""
    neg_elbo_p(M, η, samples)

MAP or "Forward KL" objective using p-samples:
    -sum(logpdf(q_dist, s for s in samples))

We simply return the negative log-likelihood w.r.t. the data.
"""
function neg_elbo_p(M, η, samples)
    # Convert to ExponentialFamily distribution in "q" form
    q = convert(ExponentialFamilyDistribution, M, η)
    q_dist = convert(Distribution, q)

    # Negative log-likelihood
    data_term = -sum(logpdf(q_dist, s) for s in eachcol(samples))
    return data_term
end

"""
    neg_elbo_q(M, η, target_dist; n_samples=2000)

Free Energy or "Reverse KL" objective, using q-samples:
    `-mean(logpdf(target_dist, s) - logpdf(q_dist, s) for s in samples)`.
"""
function neg_elbo_q(M, η, target_dist; n_samples=2000)
    # Convert to q-dist
    q = convert(ExponentialFamilyDistribution, M, η)
    q_dist = convert(Distribution, q)

    # Sample from q
    samples = rand(StableRNG(422), q_dist, n_samples)

    # Average [log p(x) - log q(x)] => negative
    log_ratios = [
        logpdf(target_dist, samples[:, i]) - logpdf(q_dist, samples[:, i])
        for i in 1:n_samples
    ]
    return -mean(log_ratios)
end

function run_free_energy(form, target_dist, η_init; max_iters=500)
    # Build manifold
    M = ExponentialFamilyManifolds.get_natural_manifold(form, (2,))

    # Objective & gradient
    obj(M, η) = neg_elbo_q(M, η, target_dist)
    grad(M, η) = ForwardDiff.gradient(x -> obj(M, x), η)

    stopping_crit = StopAfterIteration(max_iters) | StopWhenGradientNormLess(1e-6)

    η_opt = gradient_descent(M, obj, grad, η_init;
                             stopping_criterion=stopping_crit)
    convert(Distribution, convert(ExponentialFamilyDistribution, M, η_opt))
end

function run_maximum_likelihood(form, samples, η_init; max_iters=500)
    M = ExponentialFamilyManifolds.get_natural_manifold(form, (2,))
    stopping_criterion = StopAfterIteration(max_iters) | StopWhenGradientNormLess(1e-6)

    # Maximum likelihood: gather samples, do negative log-likelihood
    f(M, η) = -sum(logpdf(convert(Distribution, convert(ExponentialFamilyDistribution, M, η)), s) for s in eachcol(samples))
    g(M, η) = ForwardDiff.gradient(r -> f(M, r), η)

    η_opt = gradient_descent(M, f, g, η_init;
                             stopping_criterion=stopping_criterion)
    return convert(Distribution, convert(ExponentialFamilyDistribution, M, η_opt))
end


form = MvNormalMeanCovariance
η_init = ArrayPartition([0.4, 0.4], [-0.5 0.0; 0.0 -0.5])
target_dist = create_target_distribution()
samples = rand(StableRNG(42), target_dist, 2000)

q_maximum_likelihood = run_maximum_likelihood(form, samples, η_init)
q_full = run_free_energy(form, target_dist, η_init)

η_init_diag = ArrayPartition([0.4, 0.4], [-0.5])
q_fe_diag = run_free_energy(MvNormalMeanScalePrecision, target_dist, η_init_diag)

function plot_comparison(q1, q2, target_dist, title_q1, title_q2)
    x = range(-6, 6, length=100)
    y = range(-6, 6, length=100)

    z_target = [pdf(target_dist, [xi, yi]) for yi in y, xi in x]
    z_p      = [pdf(q1, [xi, yi]) for yi in y, xi in x]
    z_q      = [pdf(q2, [xi, yi]) for yi in y, xi in x]

    plt1 = contour(x, y, z_target; fill=true, alpha=0.4, color=:viridis,
        levels=15, title=title_q1, xlabel="x1", ylabel="x2")
    contour!(plt1, x, y, z_p; color=:red, fill=false, linewidth=2, levels=10)

    plt2 = contour(x, y, z_target; fill=true, alpha=0.4, color=:viridis,
        levels=15, title=title_q2, xlabel="x1", ylabel="x2")
    contour!(plt2, x, y, z_q; color=:blue, fill=false, linewidth=2, levels=10)

    plot(plt1, plt2, layout=(1,2), size=(1200,400))
end

plot_comparison(q_maximum_likelihood, q_full, target_dist, "Maximum likelihood", "Free Energy")
savefig("mle_vs_vi.pdf")

plot_comparison(q_fe_diag, q_full, target_dist, "Isotropic gaussian", "Gaussian") 
savefig("full_vs_isotropic.pdf")
