# AdaStress

This package implements the Adaptive Stress Testing (AST) framework, which finds the likeliest failures for a system under test.

The package provides three primary services:
- An interface between a user simulation and the AST framework
- A variety of RL-based solvers
- A suite of analysis and visualization tools

AdaStress is written in the Julia programming language. If you do not have Julia on your computer, follow [the official instructions](https://julialang.org/downloads/platform) to download the latest version. For information on using Julia, see the [language documentation](https://docs.julialang.org).

To download AdaStress, open the Julia REPL and run the command
```
]add https://babelfish.arc.nasa.gov/bitbucket/scm/adastress/adastress.git
```
Alternatively, an AdaStress docker container can be created with the dockerfile in the repository. For instructions on using AdaStress, see the [documentation](https://www.nasa.gov). For background on original AST formulation, see
> Lee, Ritchie, Ole J. Mengshoel, Anshu Saksena, Ryan W. Gardner, Daniel Genin, Joshua Silbermann, Michael Owen, and Mykel J. Kochenderfer. "Adaptive stress testing: Finding likely failure events with reinforcement learning." Journal of Artificial Intelligence Research 69 (2020): 1165-1201.