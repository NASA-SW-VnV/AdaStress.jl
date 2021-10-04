using Pkg

Pkg.add(["LocalCoverage", "TestReports"])
Pkg.develop(path=joinpath(@__DIR__, ".."))
Pkg.precompile()
