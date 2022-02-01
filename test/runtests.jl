using AdaStress
using NBInclude
using Test

@testset "Interface" begin
    @test AdaStress.GrayBox <: Interface.AbstractSimulation
end

@testset "Solvers" begin
    @test begin
        AdaStress.enable("SoftActorCritic")
        using AdaStress.SoftActorCritic
        ac = SoftActorCritic.MLPActorCritic(1, 1, [-1.0], [1.0])
        ac isa AdaStress.GlobalResult
    end
end

@testset "Analysis" begin
    @test begin
        AdaStress.enable("PolicyValueVerification")
        using AdaStress.PolicyValueVerification
        ac = SoftActorCritic.MLPActorCritic(1, 1, [-1.0], [1.0])
        SoftActorCritic.to_cpu!(ac)
        nnet = PolicyValueVerification.policy_network(ac; act_mins=[-1.0], act_maxs=[1.0])
        nnet isa PolicyValueVerification.ExtendedNetwork
    end
end

# Example notebooks to be tested
envs = ["pvv"] # ["walk1d", "walk2d", "pvv"]

for env in envs
    @testset "Example: $env" begin
        dir = joinpath(@__DIR__, "..", "examples", env)
        cd(dir) do
            @test begin
                @nbinclude(joinpath(dir, "$env.ipynb")) # regex=r"^(?!# autoskip).*$"
                true
            end
        end
    end
end
