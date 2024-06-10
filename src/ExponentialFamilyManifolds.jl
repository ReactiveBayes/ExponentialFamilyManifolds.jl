module ExponentialFamilyManifolds

using BayesBase, ExponentialFamily, ManifoldsBase, Manifolds, Random, LinearAlgebra

include("negative_definite_matrices.jl")
include("shifted_negative_numbers.jl")
include("shifted_positive_numbers.jl")
include("natural_manifolds.jl")

include("natural_manifolds/bernoulli.jl")
include("natural_manifolds/beta.jl")
include("natural_manifolds/chisq.jl")
include("natural_manifolds/dirichlet.jl")
include("natural_manifolds/exponential.jl")
include("natural_manifolds/gamma.jl")
include("natural_manifolds/geometric.jl")
include("natural_manifolds/laplace.jl")
include("natural_manifolds/lognormal.jl")
include("natural_manifolds/normal.jl")


end
