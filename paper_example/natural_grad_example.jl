"""
Natural gradient implementation as shown in Listing 4 of the paper.
"""
function natural_gradient(M, η, f)
    q = convert(ExponentialFamilyDistribution, M, η)
    F = fisherinformation(q)
    grad = ForwardDiff.gradient(x -> f(M, x), η)
    return F \ grad
end
