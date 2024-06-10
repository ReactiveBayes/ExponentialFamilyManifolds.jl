using ExponentialFamilyManifolds
using Test
using Aqua

@testset "ExponentialFamilyManifolds.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(ExponentialFamilyManifolds)
    end
    # Write your tests here.
end
