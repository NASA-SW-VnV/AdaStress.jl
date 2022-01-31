# AdaStress
-----------

This package implements the Adaptive Stress Testing (AST) framework, which determines the likeliest failures for a system under test.

The package provides three primary services:
- Interfaces between user simulations and the AST framework
- A variety of reinforcement learning-based solvers
- A suite of analysis and visualization tools

## Prerequisites

AdaStress is written in the Julia programming language. If you do not have Julia on your computer, follow [the official instructions](https://julialang.org/downloads/platform) to download the latest version. For information on using Julia, see the [language documentation](https://docs.julialang.org). To download and build the latest version of AdaStress, open the Julia REPL and run the command
```
]add https://babelfish.arc.nasa.gov/bitbucket/scm/adastress/adastress.git
```
Alternatively, a Julia/AdaStress docker container can be created from the dockerfile in the repository.

## Problem setup

To effectively make use of AdaStress, you must have a **system under test (SUT)** and a simulation that allows the SUT to interact with a **semi-stochastic environment**. In other words, there must be variables in your simulation that are random and behave according to a modeled probability distribution. The system should have at least one identifiable **criterion of failure** and you should be able to specify a measure of **distance to failure** which achieves its minimum at a failure event.

>For instance, if you are stress-testing an aircraft collision avoidance system, your simulation might involve an encounter scenario with mutiple random variables, such as pilot response, wind, and sensor noise. The criterion of failure might be two aircraft coming within a certain distance of each other. The distance to failure would then be the distance between the aircraft.

AdaStress provides two simulation interfaces, **blackbox** and **graybox**. Solvers may be compatible with one or both interfaces.
- A **blackbox** simulation does not reveal its environment variables, and performs all updates internally. AdaStress interacts with the simulation by setting a random seed.
- A **graybox** simulation makes its environment available to the solver, which can sample the random variables directly and return the values for the simulator to use in its update step. 

Your simulation should inherit from the `BlackBox` or `GrayBox` type and implement the functions in `src/interface/BlackBox.jl` or `src/interface/GrayBox.jl`, respectively. These functions are
- **`reset!`**
Resets simulation.
- **`observe`**
Returns observation of simulation (optional).
- **`step!`**
Steps simulation. If simulation is a graybox, the function takes an additional `EnvironmentValue` argument. If simulation is a blackbox, the function returns the log probability of the environment in its current state.
- **`isterminal`**
Checks whether simulation has finished, independent of SUT failure.
- **`isevent`**
Checks whether SUT is in a failure state.
- **`distance`**
Returns metric of distance to failure.

If simulation is a graybox, it must additionally implement
- **`environment`**
Returns `Environment` object constructed in simulation.

## Further information
For more detailed instructions on using AdaStress, see the [complete documentation](https://www.nasa.gov). Example notebooks can be found in the `examples` directory. For background on original AST formulation, see
> Lee, Ritchie, Ole J. Mengshoel, Anshu Saksena, Ryan W. Gardner, Daniel Genin, Joshua Silbermann, Michael Owen, and Mykel J. Kochenderfer. "Adaptive stress testing: Finding likely failure events with reinforcement learning." Journal of Artificial Intelligence Research 69 (2020): 1165-1201.

## License
AdaStress has been released under the NASA Open Source Agreement version 1.3, as detailed [here](LICENSE.pdf).

## Maintainers
- Rory Lipkis (`rory.lipkis@nasa.gov`)
- Adrian Agogino (`adrian.k.agogino@nasa.gov`)
