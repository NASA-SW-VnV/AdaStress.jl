using Pkg

Pkg.add([
    PackageSpec(name="LocalCoverage"),
    PackageSpec(name="TestReports"),
    PackageSpec(url="https://github.com/sisl/NeuralVerification.jl"),
    PackageSpec(path=joinpath(@__DIR__, ".."))
])
