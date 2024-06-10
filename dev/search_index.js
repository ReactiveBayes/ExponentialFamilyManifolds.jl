var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = ExponentialFamilyManifolds","category":"page"},{"location":"#ExponentialFamilyManifolds","page":"Home","title":"ExponentialFamilyManifolds","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"ExponentialFamilyManifolds.jl provides implementations of manifolds for the natural parameters of exponential family distributions, using Manifolds.jl. These manifolds are compatible with ManifoldsBase.jl, enabling optimization of the natural parameters of exponential family distributions using Manopt.jl.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"#Manifolds","page":"Home","title":"Manifolds","text":"","category":"section"},{"location":"#Distribution-specific-manifolds","page":"Home","title":"Distribution specific manifolds","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"ExponentialFamilyManifolds.get_natural_manifold\nExponentialFamilyManifolds.NaturalParametersManifold\nExponentialFamilyManifolds.get_natural_manifold_base\nExponentialFamilyManifolds.partition_point","category":"page"},{"location":"#ExponentialFamilyManifolds.get_natural_manifold","page":"Home","title":"ExponentialFamilyManifolds.get_natural_manifold","text":"get_natural_manifold(::Type{T}, dims, conditioner = nothing)\n\nThe function returns a corresponding manifold for the natural parameters of distribution of type T. Optionally accepts the conditioner, which is set to nothing by default. Use empty tuple () for univariate distributions. \n\njulia> using ExponentialFamily, ExponentialFamilyManifolds\n\njulia> ExponentialFamilyManifolds.get_natural_manifold(Beta, ()) isa ExponentialFamilyManifolds.NaturalParametersManifold\ntrue\n\njulia> ExponentialFamilyManifolds.get_natural_manifold(MvNormalMeanCovariance, (3, )) isa ExponentialFamilyManifolds.NaturalParametersManifold\ntrue\n\n\n\n\n\n","category":"function"},{"location":"#ExponentialFamilyManifolds.NaturalParametersManifold","page":"Home","title":"ExponentialFamilyManifolds.NaturalParametersManifold","text":"NaturalParametersManifold(::Type{T}, dims, base, conditioner)\n\nThe manifold for the natural parameters of the distribution of type T with dimensions dims. An internal structure, use get_natural_manifold to create an instance of a manifold for the natural parameters of distribution of type T.\n\n\n\n\n\n","category":"type"},{"location":"#ExponentialFamilyManifolds.get_natural_manifold_base","page":"Home","title":"ExponentialFamilyManifolds.get_natural_manifold_base","text":"get_natural_manifold_base(M::NaturalParametersManifold)\nget_natural_manifold_base(::Type{T}, dims, conditioner = nothing)\n\nReturns base manifold for the distribution of type T of dimension dims. Optionally accepts the conditioner, which is set to nothing by default.\n\n\n\n\n\nget_natural_manifold_base(::Type{Bernoulli}, ::Tuple{}, conditioner=nothing)\n\nGet the natural manifold base for the Bernoulli distribution.\n\n\n\n\n\nget_natural_manifold_base(::Type{Beta}, ::Tuple{}, conditioner=nothing)\n\nGet the natural manifold base for the Beta distribution.\n\n\n\n\n\nget_natural_manifold_base(::Type{Chisq}, ::Tuple{}, conditioner=nothing)\n\nGet the natural manifold base for the Chisq distribution.\n\n\n\n\n\nget_natural_manifold_base(::Type{Dirichlet}, dims::Tuple{Int}, conditioner=nothing)\n\nGet the natural manifold base for the Dirichlet distribution.\n\n\n\n\n\nget_natural_manifold_base(::Type{Exponential}, ::Tuple{}, conditioner=nothing)\n\nGet the natural manifold base for the Exponential distribution.\n\n\n\n\n\nget_natural_manifold_base(::Type{Gamma}, ::Tuple{}, conditioner=nothing)\n\nGet the natural manifold base for the Gamma distribution.\n\n\n\n\n\nget_natural_manifold_base(::Type{Geometric}, ::Tuple{}, conditioner=nothing)\n\nGet the natural manifold base for the Geometric distribution.\n\n\n\n\n\nget_natural_manifold_base(::Type{Laplace}, ::Tuple{}, conditioner=nothing)\n\nGet the natural manifold base for the Laplace distribution.\n\n\n\n\n\nget_natural_manifold_base(::Type{LogNormal}, ::Tuple{}, conditioner=nothing)\n\nGet the natural manifold base for the LogNormal distribution.\n\n\n\n\n\nget_natural_manifold_base(::Type{NormalMeanVariance}, ::Tuple{}, conditioner = nothing)\n\nGet the natural manifold base for the NormalMeanVariance distribution.\n\n\n\n\n\nget_natural_manifold_base(::Type{MvNormalMeanCovariance}, dims::Tuple{Int}, conditioner = nothing)\n\nGet the natural manifold base for the MvNormalMeanCovariance distribution.\n\n\n\n\n\n","category":"function"},{"location":"#ExponentialFamilyManifolds.partition_point","page":"Home","title":"ExponentialFamilyManifolds.partition_point","text":"partition_point(M::NaturalParametersManifold, p)\npartition_point(::Type{T}, dims, point, conditioner=nothing)\n\nConverts the point to a compatible representation for the natural manifold M of type T.\n\n\n\n\n\npartition_point(::Type{Bernoulli}, ::Tuple{}, p, conditioner=nothing)\n\nConverts the point to a compatible representation for the natural manifold of type Bernoulli.\n\n\n\n\n\npartition_point(::Type{Beta}, ::Tuple{}, p, conditioner=nothing)\n\nConverts the point to a compatible representation for the natural manifold of type Beta.\n\n\n\n\n\npartition_point(::Type{Chisq}, ::Tuple{}, p, conditioner=nothing)\n\nConverts the point to a compatible representation for the natural manifold of type Chisq.\n\n\n\n\n\npartition_point(::Type{Dirichlet}, dims::Tuple{Int}, p, conditioner=nothing)\n\nConverts the point to a compatible representation for the natural manifold of type Dirichlet.\n\n\n\n\n\npartition_point(::Type{Exponential}, ::Tuple{}, p, conditioner=nothing)\n\nConverts the point to a compatible representation for the natural manifold of type Exponential.\n\n\n\n\n\npartition_point(::Type{Gamma}, ::Tuple{}, p, conditioner=nothing)\n\nConverts the point to a compatible representation for the natural manifold of type Gamma.\n\n\n\n\n\npartition_point(::Type{Geometric}, ::Tuple{}, p, conditioner=nothing)\n\nConverts the point to a compatible representation for the natural manifold of type Geometric.\n\n\n\n\n\npartition_point(::Type{Laplace}, ::Tuple{}, p, conditioner=nothing)\n\nConverts the point to a compatible representation for the natural manifold of type Laplace.\n\n\n\n\n\npartition_point(::Type{LogNormal}, ::Tuple{}, p, conditioner=nothing)\n\nConverts the point to a compatible representation for the natural manifold of type LogNormal.\n\n\n\n\n\npartition_point(::Type{NormalMeanVariance}, ::Tuple{}, p, conditioner = nothing)\n\nConverts the point to a compatible representation for the natural manifold of type NormalMeanVariance.\n\n\n\n\n\npartition_point(::Type{MvNormalMeanCovariance}, dims::Tuple{Int}, p, conditioner = nothing)\n\nConverts the point to a compatible representation for the natural manifold of type MvNormalMeanCovariance.\n\n\n\n\n\n","category":"function"},{"location":"#Custom-generic-manifolds","page":"Home","title":"Custom generic manifolds","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"ExponentialFamilyManifolds.jl provides some extra manifolds, which are not included in the Manifolds.jl","category":"page"},{"location":"","page":"Home","title":"Home","text":"ExponentialFamilyManifolds.ShiftedPositiveNumbers\nExponentialFamilyManifolds.ShiftedNegativeNumbers\nExponentialFamilyManifolds.SymmetricNegativeDefinite","category":"page"},{"location":"#ExponentialFamilyManifolds.ShiftedPositiveNumbers","page":"Home","title":"ExponentialFamilyManifolds.ShiftedPositiveNumbers","text":"ShiftedPositiveNumbers(shift)\n\nA manifold representing the positive numbers shifted by shift.  The points on this manifold are 1-dimensional vectors with a single element.\n\n\n\n\n\n","category":"type"},{"location":"#ExponentialFamilyManifolds.ShiftedNegativeNumbers","page":"Home","title":"ExponentialFamilyManifolds.ShiftedNegativeNumbers","text":"ShiftedNegativeNumbers(shift)\n\nA manifold representing the negative numbers shifted by shift. The points on this manifold are 1-dimensional vectors with a single element.\n\n\n\n\n\n","category":"type"},{"location":"#ExponentialFamilyManifolds.SymmetricNegativeDefinite","page":"Home","title":"ExponentialFamilyManifolds.SymmetricNegativeDefinite","text":"SymmetricNegativeDefinite(k)\n\nThis manifold represents the set of negative definite matrices of size k × k.  Similar to SymmetricPositiveDefinite from Manifolds.jl with the exception that the matrices are negative definite.\n\n\n\n\n\n","category":"type"},{"location":"#Helpers","page":"Home","title":"Helpers","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"ExponentialFamilyManifolds.Negated","category":"page"},{"location":"#ExponentialFamilyManifolds.Negated","page":"Home","title":"ExponentialFamilyManifolds.Negated","text":"Negated(m)\n\nLazily negates the matrix m, without creating a new matrix.  Works by redefining the getindex.\n\n\n\n\n\n","category":"type"}]
}
