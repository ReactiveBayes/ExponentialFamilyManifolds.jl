module ExponentialFamilyManifolds

using BayesBase, ExponentialFamily, ManifoldsBase, Manifolds, Random, LinearAlgebra

include("symmetric_negative_definite.jl")
include("shifted_negative_numbers.jl")
include("shifted_positive_numbers.jl")
include("natural_manifolds.jl")

include("natural_manifolds/bernoulli.jl")
include("natural_manifolds/beta.jl")
include("natural_manifolds/binomial.jl")
include("natural_manifolds/chisq.jl")
include("natural_manifolds/categorical.jl")
include("natural_manifolds/dirichlet.jl")
include("natural_manifolds/exponential.jl")
include("natural_manifolds/gamma.jl")
include("natural_manifolds/geometric.jl")
include("natural_manifolds/laplace.jl")
include("natural_manifolds/lognormal.jl")
include("natural_manifolds/normal.jl")
include("natural_manifolds/negative_binomial.jl")
include("natural_manifolds/pareto.jl")
include("natural_manifolds/poisson.jl")
include("natural_manifolds/rayleigh.jl")
include("natural_manifolds/weibull.jl")
include("natural_manifolds/wishart.jl")
end
