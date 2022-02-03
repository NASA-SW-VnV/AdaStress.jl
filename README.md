![AdaStress](docs/logo.svg)

AdaStress is a software package that implements the Adaptive Stress Testing (AST) framework, which determines the likeliest failures for a system under test.

AdaStress provides three primary services:
- Interfaces between user simulations and the AST framework
- Reinforcement learning-based solvers
- Analysis and visualization tools

## Prerequisites

AdaStress is written in the Julia programming language. If you do not have Julia on your computer, follow [the official instructions](https://julialang.org/downloads/platform) to download the latest version. For information on using Julia, see the [language documentation](https://docs.julialang.org). To download and build the latest version of AdaStress, open the Julia REPL, type `]` to enter the interactive package mode, and enter the command
```
add https://babelfish.arc.nasa.gov/bitbucket/scm/adastress/adastress.git
```
To use the package, enter the command
```
using AdaStress
```

## Problem setup

To effectively make use of AdaStress, you must have a **system under test (SUT)** and a **simulation** in which the SUT interacts with a **semi-stochastic environment**. This means that there should be variables in your simulation that are random and behave according to a modeled probability distribution. The system should have an identifiable **failure criterion** and you should preferably be able to specify a measure of **distance to failure**, a scalar quantity which achieves its minimum at a failure event.

>For instance, if you are stress-testing an aircraft collision avoidance system, your simulation might involve an encounter scenario with multiple random variables, such as pilot intent, wind, and sensor noise. The failure criterion might be the condition of two aircraft coming within a certain distance of each other. The distance to failure would then be the instantaneous distance between the aircraft. For an involved implementation of this problem, see the example in `examples/cas`.

AdaStress provides two basic simulation interfaces, **black-box** and **gray-box**. The type of simulation determines which solvers are may be used.

- A **black-box** simulation does not reveal its environment variables and performs all updates internally. AdaStress interacts with the simulation by setting a random seed.
- A **gray-box** simulation makes its environment available to the solver, which can sample the random variables directly and return the values for the simulator to use in its update step. 

Your simulation must inherit from the `BlackBox` or `GrayBox` type and implement the methods found in `src/interface/BlackBox.jl` or `src/interface/GrayBox.jl`.

## Further information
For more detailed instructions on using AdaStress, see the [complete documentation](./docs/main.md). Example notebooks can be found in the `examples` directory. For background on original AST formulation, see
> Lee, Ritchie, Ole J. Mengshoel, Anshu Saksena, Ryan W. Gardner, Daniel Genin, Joshua Silbermann, Michael Owen, and Mykel J. Kochenderfer. "Adaptive stress testing: Finding likely failure events with reinforcement learning." Journal of Artificial Intelligence Research 69 (2020): 1165-1201.

## License
AdaStress has been released under the NASA Open Source Agreement version 1.3, as detailed [here](LICENSE.pdf).

## Maintainers
- Rory Lipkis (`rory.lipkis@nasa.gov`)
- Adrian Agogino (`adrian.k.agogino@nasa.gov`)
