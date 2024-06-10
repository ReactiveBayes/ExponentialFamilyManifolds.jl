using ExponentialFamilyManifolds
using Documenter

DocMeta.setdocmeta!(ExponentialFamilyManifolds, :DocTestSetup, :(using ExponentialFamilyManifolds); recursive=true)

makedocs(;
    modules=[ExponentialFamilyManifolds],
    authors="Bagaev Dmitry <bvdmitri@gmail.com> and contributors",
    sitename="ExponentialFamilyManifolds.jl",
    format=Documenter.HTML(;
        canonical="https://ReactiveBayes.github.io/ExponentialFamilyManifolds.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ReactiveBayes/ExponentialFamilyManifolds.jl",
    devbranch="main",
)
