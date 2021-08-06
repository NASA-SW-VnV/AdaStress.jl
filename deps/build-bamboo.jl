ENV["PYTHON"] = "" # prompts internal Conda installation
using Pkg

Pkg.add([
    PackageSpec(name="PyCall"),
    PackageSpec(name="PyPlot"),
    PackageSpec(name="LocalCoverage"),
    PackageSpec(name="TestReports"),
    PackageSpec(url="https://github.com/sisl/NeuralVerification.jl")
])

Pkg.develop(path=joinpath(@__DIR__, ".."))
Pkg.precompile()
