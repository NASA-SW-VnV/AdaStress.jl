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

test_envs = []
for env in test_envs
    @testset "Example: $env" begin
        @test @nbinclude("$env.ipynb")
    end
end
