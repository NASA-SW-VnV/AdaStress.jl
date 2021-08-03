using AdaStress
using Test

@testset "AdaStress.jl" begin
    @test AdaStress.GrayBox <: AdaStress.Interface.AbstractSimulation # trivial
end
