@testitem "Aqua: Auto QUality Assurance" begin
    using Aqua, ExponentialFamilyManifolds

    Aqua.test_all(
        ExponentialFamilyManifolds;
        ambiguities=false,
        deps_compat=(; check_extras=false, check_weakdeps=true),
    )
end