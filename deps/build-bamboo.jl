using Pkg

Pkg.add([
    PackageSpec(name="LocalCoverage"),
    PackageSpec(name="TestReports"),
    PackageSpec(url="https://github.com/sisl/NeuralVerification.jl")
])

Pkg.develop(path=joinpath(@__DIR__, ".."))
Pkg.precompile()
