using Pkg

Pkg.add([
    PackageSpec(name="LocalCoverage"),
    PackageSpec(name="TestReports")
])

Pkg.develop(path=joinpath(@__DIR__, ".."))
Pkg.precompile()
