import LocalCoverage
import Pkg
using TestReports

testdir = "test-reports"
cd(mkpath(testdir))

Pkg.test(p::Symbol; kwargs...) = TestReports.test(String(p); kwargs...)
LocalCoverage.pkgdir(p::Symbol) = LocalCoverage.pkgdir(String(p))
LocalCoverage.generate_coverage(:AdaStress)

