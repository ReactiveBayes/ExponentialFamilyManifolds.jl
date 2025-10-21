using Aqua, Hwloc, ExponentialFamilyManifolds, Test, ReTestItems

Aqua.test_all(
    ExponentialFamilyManifolds;
    ambiguities=false,
    deps_compat=(; check_extras=false, check_weakdeps=true),
    piracies=false,
)

ncores = max(Hwloc.num_physical_cores(), 1)
nthreads = max(Hwloc.num_virtual_cores(), 1)
threads_per_core = max(Int(floor(nthreads / ncores)), 1)

runtests(
    ExponentialFamilyManifolds;
    nworkers=ncores,
    nworker_threads=threads_per_core,
    memory_threshold=1.0,
)
