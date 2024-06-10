```@meta
CurrentModule = ExponentialFamilyManifolds
```

# ExponentialFamilyManifolds

`ExponentialFamilyManifolds.jl` provides implementations of manifolds for the natural parameters of exponential family distributions, using `Manifolds.jl`. These manifolds are compatible with `ManifoldsBase.jl`, enabling optimization of the natural parameters of exponential family distributions using `Manopt.jl`.

```@index
```

# Manifolds

## Custom generic manifolds

`ExponentialFamilyManifolds.jl` provides some extra manifolds, which are not included in the `Manifolds.jl`

```@docs
ExponentialFamilyManifolds.ShiftedPositiveNumbers
ExponentialFamilyManifolds.ShiftedNegativeNumbers
ExponentialFamilyManifolds.SymmetricNegativeDefinite
```

## Distribution specific manifolds

# Helpers 

```@docs 
ExponentialFamilyManifolds.Negated
```