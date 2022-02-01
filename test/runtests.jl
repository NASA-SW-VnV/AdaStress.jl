# ******************************************************************************************
# Notices:
#
# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.
#
# Disclaimers
#
# No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF ANY KIND,
# EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY
# THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR FREEDOM FROM INFRINGEMENT, ANY
# WARRANTY THAT THE SUBJECT SOFTWARE WILL BE ERROR FREE, OR ANY WARRANTY THAT DOCUMENTATION,
# IF PROVIDED, WILL CONFORM TO THE SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT, IN ANY MANNER,
# CONSTITUTE AN ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS,
# RESULTING DESIGNS, HARDWARE, SOFTWARE PRODUCTS OR ANY OTHER APPLICATIONS RESULTING FROM
# USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY DISCLAIMS ALL WARRANTIES AND
# LIABILITIES REGARDING THIRD-PARTY SOFTWARE, IF PRESENT IN THE ORIGINAL SOFTWARE, AND
# DISTRIBUTES IT "AS IS."
#
# Waiver and Indemnity:  RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE UNITED
# STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT.
# IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS IN ANY LIABILITIES, DEMANDS, DAMAGES,
# EXPENSES OR LOSSES ARISING FROM SUCH USE, INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON,
# OR RESULTING FROM, RECIPIENT'S USE OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND
# HOLD HARMLESS THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL
# AS ANY PRIOR RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR ANY
# SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS AGREEMENT.
# ******************************************************************************************

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

# Example notebooks to be tested (cells beginning with `# autoskip` are not executed)
envs = ["walk1d", "walk2d", "pvv"]

for env in envs
    @testset "Example: $env" begin
        dir = joinpath(@__DIR__, "..", "examples", env)
        cd(dir) do
            @test begin
                @nbinclude(joinpath(dir, "$env.ipynb"); regex=r"^(?!# autoskip).*")
                true
            end
        end
    end
end
