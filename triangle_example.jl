using Distributions, LinearAlgebra, Random, Plots, RecursiveArrayTools
using ExponentialFamily, ExponentialFamilyManifolds
using Manopt, ForwardDiff, StableRNGs
using Statistics

############################################################################
# 1) Define a bimodal (actually tri-modal) target distribution
############################################################################
function create_target_distribution()
    # Coordinates for an equilateral triangle:
    mu1 = [0.0, 0.0]
    mu2 = [2.0, 0.0]
    mu3 = [1.0, 1.732]   # sqrt(3) ~ 1.732

    # Covariance for each mode
    Σ = 0.2 .* Matrix(I, 2, 2)
    w = [1/3, 1/3, 1/3]

    return MixtureModel([
        MvNormal(mu1, Σ),
        MvNormal(mu2, Σ),
        MvNormal(mu3, Σ),
    ], w)
end

############################################################################
# 2) Forward KL objective: negative log-likelihood (no explicit prior)
############################################################################

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

############################################################################
# 3) Reverse KL objective: E_q[ log q(x) - log p(x) ]
############################################################################
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

############################################################################
# Utility to run "Free Energy" approach
############################################################################
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

############################################################################
# 4) Main: run forward (MAP) and reverse (FE) KL
############################################################################
function run_forward_vs_reverse(form, η_init; max_iters=100)
    target_dist = create_target_distribution()
    M = ExponentialFamilyManifolds.get_natural_manifold(form, (2,))
    stopping_criterion = StopAfterIteration(max_iters) | StopWhenGradientNormLess(1e-6)

    # Forward KL: gather p-samples, do negative log-likelihood
    samples_p = rand(StableRNG(42), target_dist, 2000)
    f_p(M, p) = neg_elbo_p(M, p, samples_p)
    g_p(M, p) = ForwardDiff.gradient(r -> f_p(M, r), p)

    η_opt_p = gradient_descent(M, f_p, g_p, η_init;
                               stopping_criterion=stopping_criterion)
    q_p = convert(Distribution, convert(ExponentialFamilyDistribution, M, η_opt_p))

    # Reverse KL:
    q_q = run_free_energy(form, target_dist, η_init; max_iters=max_iters)

    return q_p, q_q, target_dist
end

############################################################################
# 6) Plot: target vs q_p (Forward KL) vs q_q (Reverse KL)
############################################################################
function plot_comparison(q_p, q_q, target_dist)
    x = range(-6, 6, length=100)
    y = range(-6, 6, length=100)

    z_target = [pdf(target_dist, [xi, yi]) for yi in y, xi in x]
    z_p      = [pdf(q_p, [xi, yi]) for yi in y, xi in x]
    z_q      = [pdf(q_q, [xi, yi]) for yi in y, xi in x]

    plt1 = contour(x, y, z_target; fill=true, alpha=0.4, color=:viridis,
        levels=15, title="Target + Forward KL (q_p)", xlabel="x1", ylabel="x2")
    contour!(plt1, x, y, z_p; color=:red, fill=false, linewidth=2, levels=10)

    plt2 = contour(x, y, z_target; fill=true, alpha=0.4, color=:viridis,
        levels=15, title="Target + Reverse KL (q_q)", xlabel="x1", ylabel="x2")
    contour!(plt2, x, y, z_q; color=:blue, fill=false, linewidth=2, levels=10)

    plot(plt1, plt2, layout=(1,2), size=(1200,400))
end

############################################################################
# 5) Example usage
############################################################################
form = MvNormalMeanCovariance
η_init = ArrayPartition([0.4, 0.4], [-0.5 0.0; 0.0 -0.5])
q_p_map, q_q, target_dist = run_forward_vs_reverse(form, η_init)

# Print final distributions
println("MAP (Forward KL) q_p = ", q_p_map)
println("Reverse KL      q_q = ", q_q)

println("KL(q_p_map || target) = ", kldivergence(q_p_map, target_dist))
println("KL(q_q     || target) = ", kldivergence(q_q, target_dist))

# Show Forward (red) vs. Reverse (blue) KL fits
plot_comparison(q_p_map, q_q, target_dist)

# Compare restricted vs. full
η_init_diag = ArrayPartition([0.4, 0.4], [-0.5])
q_fe_diag = run_free_energy(MvNormalMeanScalePrecision, target_dist, η_init_diag)

println("q_fe_diag = ", q_fe_diag)
println("KL(q_fe_diag || target) = ", kldivergence(q_fe_diag, target_dist))
plot_comparison(q_fe_diag, q_q, target_dist)