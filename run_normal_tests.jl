#!/usr/bin/env julia

# Minimal script to run Normal manifold tests outside the test suite

using Pkg
Pkg.activate(".")

# Load the package
using ExponentialFamilyManifolds

# Load test dependencies
using Test
using StableRNGs
using ExponentialFamily
using ManifoldsBase
using LinearAlgebra
using Distributions
using Random
using Manopt
using ManifoldDiff
import ADTypes: AutoForwardDiff
import Distributions: kldivergence, Distribution
import ExponentialFamilyManifolds: get_fisher_manifold, partition_point
import ManifoldDiff: TangentDiffBackend

# Include setup test functions
include("test/metric_manifolds/metric_manifolds_setuptests.jl")
include("test/metric_manifolds/mle_metric_manifolds_setuptests.jl")

println("=" ^ 70)
println("Testing Normal metric manifold")
println("=" ^ 70)

@testset "Normal metric manifold" begin
    test_metric_manifold(tol=1e-5, maximal_norm=0.3) do rng
        return NormalMeanVariance(10randn(rng), rand(rng) + 1)
    end
end

println("\n" * "=" ^ 70)
println("Testing Normal fisher manifold MLE")
println("=" ^ 70)

@testset "Normal fisher manifold MLE" begin
    test_mle_works() do rng
        return NormalMeanVariance(10randn(rng), 10rand(rng))
    end
end

println("\n" * "=" ^ 70)
println("All tests completed!")
println("=" ^ 70)

