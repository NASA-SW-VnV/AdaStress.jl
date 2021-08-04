ENV["PYTHON"] = "" # prompts internal Conda installation
using Pkg

Pkg.add("PyCall")
Pkg.build("PyCall") # force Conda rebuild

Pkg.add([
    PackageSpec(name="PyPlot"),
    PackageSpec(name="TestReports"),
    PackageSpec(url="https://github.com/sisl/NeuralVerification.jl")
])

Pkg.develop(path=joinpath(@__DIR__, ".."))
Pkg.precompile()
