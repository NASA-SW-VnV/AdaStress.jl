![AdaStress](../logo.svg)

# Documentation

## Maintainers
- Rory Lipkis (`rory.lipkis@nasa.gov`)
- Adrian Agogino (`adrian.k.agogino@nasa.gov`)

## Description

AdaStress is a software package that implements the [Adaptive Stress Testing (AST) framework](https://doi.org/10.1613/jair.1.12190), which determines the likeliest failures for a system under test.

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

## Architecture
AdaStress is designed with an emphasis on speed and modularity. The architecture is partitioned into several high-level modules that interact through common interfaces. This allows the software to be extended with new modules offering additional functionality without altering the base code.

A custom simulation interacts with the `Interface` module to product an `ASTMDP` representing the AST formulation of the problem. The MDP interacts with the `Solver` module to produce a `Result`. A result object can optionally be passed into the `Analysis` module to produce various artifacts. 

## Problem setup

To effectively make use of AdaStress, you must have a **system under test (SUT)** and a **simulation** in which the SUT interacts with a **semi-stochastic environment**. This means that there should be variables in your simulation that are random and behave according to a modeled probability distribution. The system should have an identifiable **failure criterion** and you should preferably be able to specify a measure of **distance to failure**, a scalar quantity which achieves its minimum at a failure event.

>For instance, if you are stress-testing an aircraft collision avoidance system, your simulation might involve an encounter scenario with multiple random variables, such as pilot intent, wind, and sensor noise. The failure criterion might be the condition of two aircraft coming within a certain distance of each other. The distance to failure would then be the instantaneous distance between the aircraft. For an involved implementation of this problem, see the example in `examples/cas`.

AdaStress provides two basic simulation interfaces, **black-box** and **gray-box**. The type of simulation determines which solvers are may be used.

- A **black-box** simulation does not reveal its environment variables and performs all updates internally. AdaStress interacts with the simulation by setting a random seed.
- A **gray-box** simulation makes its environment available to the solver, which can sample the random variables directly and return the values for the simulator to use in its update step. 

Your simulation must inherit from the `BlackBox` or `GrayBox` type and implement the methods found in `src/interface/BlackBox.jl` or `src/interface/GrayBox.jl`. These methods are
> - **`AdaStress.reset!`**
> Resets simulation.
> - **`AdaStress.observe`**
> Returns observation of simulation (optional).
> - **`AdaStress.step!`**
> Steps simulation. If simulation is a gray-box, the function takes an additional `EnvironmentValue` argument. If simulation is a black-box, the function returns the log probability of the environment in its current state.
> - **`AdaStress.isterminal`**
> Checks whether simulation has finished, independent of SUT failure.
> - **`AdaStress.isevent`**
> Checks whether SUT is in a failure state.
> - **`AdaStress.distance`**
> Returns distance to failure.

If simulation is a gray-box, it must additionally implement
> - **`environment`**
> Returns `Environment` object constructed in simulation.

The `Environment` type is an alias of `Dict{Symbol, Sampleable}`, i.e., a dictionary mapping symbols to sampleable objects, typically probability distributions. The environment contains the stochastic variables in your simulation that you intend for AdaStress to recognize. This can be simple, as in
```
using Distributions
AdaStress.environment(sim::ExampleSim) = Dict(:wind => Normal(0.0, 0.05))
```
or arbitrarily complicated and constructed programmatically. To avoid unnecessary reallocation, it is preferable for the environment to be constructed once and stored in the simulation object, wjere it can be updated as necessary. In this way, the `environment` function simply references the variable, as in
```
AdaStress.environment(sim::ExampleSim) = sim.environment
```
The `EnvironmentValue` type is an alias of `Dict{Symbol, Any}`. An environment value object corresponds to an environment object in which each variable has been sampled. These objects are never constructed by the user; rather, they are generated by AdaStress and passed into your simulation for the update step, as in the following example:
```
function AdaStress.step!(sim::ExampleSim, x::EnvironmentValue)
    [...]
    system_reponse = system_under_test(sim.state, x[:wind])
    [...]
end
```
For simple examples of black-box and gray-box simulations, see the notebooks in `examples/walk1d` and `examples/walk2d`, respectively.

If your distribution is uncommon or custom-written, you must implement functions to allow the flattening and unflattening of samples. The flattening process should pseudo-normalize the samples if possible, as in the following example:
```
struct CustomDistribution <: Distribution{Multivariate, Continuous}
    a::Float64
    b::Float64
end

struct CustomValue
    c::Float64
    d::Float64
end

Base.rand(d::CustomDistribution) = [...] # complicated sampling scheme; returns a CustomValue

AdaStress.flatten(d::CustomDistribution, v::CustomValue) = [v.c / d.a, v.d / d.b]
AdaStress.unflatten(d::CustomDistribution, v::Vector{Float64}) = CustomValue(v[1] * d.a, v[2] * d.b)
```
For a less contrived example of flattening, see the example in `examples/cas`. For typical univariate continuous distributions, AdaStress attempts to automatically infer the flattening scheme; this can be overridden in your code if desired.

If you require a custom addition to the AST reward function, you may optionally implement the function `AdaStress.reward`, which takes the same arguments as `AdaStress.step!` and returns a scalar value or a callback to include the post-step simulation. However, use of this function is discouraged, as the built-in reward options are typically sufficient, and improperly constructed reward functions can destabilize learning and/or negate the AST formulation. This is discussed further in the next section.

## Interface

Once a simulation adheres to the gray-box or black-box interface, it can be converted into a fully-defined problem to interface with solvers. This is accomplished by wrapping an instance of the simulation in an `ASTMDP`, as in
```
problem = ASTMDP(ExampleSim())
```
Various AST parameters can be specified and changed during construction. 

### Rewards

In additional to wrapping the simulation, the `ASTMDP` holds a reward object, which specifies the reward structure to be used. Several parameters define the reward:

| Parameter | Type | Default | Description |
| - | - | - | - |
| `marginalize` | `Bool` | `true` | Determine whether environment distributions are automatically marginalized when log probabilities are computed. Marginalization requires that the distribution implement `Distributions.mode`, which is typically the case. |
| `heuristic` | `AbstractDistanceHeuristic` | `GradientHeuristic()` | Determines how distance metric is interpreted. |
| `event_bonus` | `Float64` | `0.0` | Reward bonus for encountering a failure. |
| `reward_function`| `AbstractCoreObjective` | `WeightedObjective()` | Defines how probability, distance heuristic, and event bonus are combined. Alternative values are highly experimental. |

It is common to specify reward parameters, and the `ASTMDP` constructor allows mixing keyword arguments from different levels, automatically assigning them to the inner `Reward` object if they do not match a field in the outer `ASTMDP` object. Therefore, the event bonus can be set via explicit modification, as in
```
problem = ASTMDP(ExampleSim())
problem.reward.event_bonus = 1000.0
```
or, more concisely, as in
```
problem = ASTMDP(ExampleSim(); reward_bonus=1000.0)
```

> A general rule of thumb is that the event bonus should be high enough to "outweigh" the maximum cumulative unlikeliness. In the case of a normally distributed environment disturbance, a 5-sigma value corresponds to a marginal log probability of -12.5. If your simulation is 30 steps, the cumulative log likelihood is -375. Therefore, setting the event bonus to 375.0 could be roughly interpreted as permitting up to 5-sigma of unlikeliness at every possible timestep if it leads to a failure event. Since the theoretical AST framework views failure as a constraint, rather than an objective, it is preferable to permit *any* degree of unlikeliness, within reason.

### Heuristics

Different distance heuristics can also be set, depending on the problem:
- `GradientHeuristic`: Gradient of distance metric. Default and recommended.
- `MinimumHeuristic`: Minimum distance across episode. Non-Markovian.
- `FinalHeuristic`: Distance at episode termination. Recommended (and automatically inferred) for episodic problems.

### Episodic and non-episodic problems

Another parameter of the `ASTMDP` is the Boolean value `episodic`, which defaults to `false`. Episodic problems constitute a relatively rare case in which the condition of failure and the distance to failure can only be evaluated after a set number of steps. In this case, the simulation interface changes slightly: `isevent` should check whether failure occurred at any point during the episode and `distance` should return the minimum distance to failure seen across the entire episode (i.e., the "miss distance"). This may require additional simulation-side bookkeeping.

For an example of an episodic problem, see the notebook `examples/fms`.

## Serialization interface

In certain situations, it may be useful to run the testing and simulation as separate independent processes.  AdaStress supports this mode of operation via an optional serialization interface between the simulation and the solver. There are three major use cases for this feature:

### Restricting information flow 

In some cases, it may be required to keep a separation between the simulation and tester.  The serialization interface can be used to enforce the restriction of information from the simulation to the tester. 

### Distributed computing

The serialization interface also permits distributed computing, allowing stress testing to scale across multiple processors.

### Cross-language support

The serialization capabilities also make it easier to interact with other programming languages in cases where there does not exist a robust Julia interface. Instead of reimplementing the full package in another language, it is only strictly necessary to implement the interface side. A related use case is bridging incompatible versions of Julia.

## Submodule management

The submodule manager allows optional features with heavy dependencies to be made available without increasing the loading time of the base package. The user can selectively enable and disable these submodules as needed. In the background, the submodule manager maintains an internal project environment with a minimal set of necessary dependencies, avoiding the need to precompile unused packages.

### Using submodules

Submodules are managed through the following API:

> - **`AdaStress.submodules()`**
> List all available submodules.
> - **`AdaStress.enabled()`**
> List enabled submodules.
> - **`AdaStress.enable(submodule)`**
> Enable submodule(s). Accepts string or vector of strings. With zero arguments defaults to all associated submodules. Takes effect immediately.
> - **`AdaStress.disable(submodule)`**
> Disable submodule(s). Accepts string or vector of strings. With zero arguments defaults to all enabled submodules. Takes effect after Julia restart.
> - **`AdaStress.load()`**
> Load enabled submodules (necessary after Julia restart). Takes effect immediately.
> - **`AdaStress.clean()`**
> Forcibly remove temporary environment, purging all enabled submodules. Only necessary if submodule manager is corrupted and `disable` cannot restore functionality. Takes effect after Julia restart.

Enabling a submodule can take several seconds, particularly the first time. Due to current limitations of the language, previously enabled submodules are not automatically loaded when a new Julia session is launched. The user should use the `load` command for this, as in the following example. In the first session, it is necessary to run

> ```
> julia> using AdaStress
> julia> AdaStress.enable("SoftActorCritic")
> ```
while in later sessions, the user may simply run
> ```
> julia> using AdaStress
> julia> AdaStress.load()
> ```

### Creating submodules

Custom submodules are essentially regular Julia packages that reside within the AdaStress directory tree, complete with a UUID and `Project.toml` file. Submodules are associated with AdaStress via the `exclude` command, similarly to how source files are associated via `include`.

## Solvers

A solver object is a standalone entity representing an algorithm and its particular parameter. A solver can be applied to an `ASTMDP` or a function that generates an `ASTMDP`, producing a `Result` object, as in
```
mdp = ASTMDP(ExampleSim())
solver = ExampleSolver(kwargs...)
sol = solver(mdp)
```
AdaStress offers two families of solvers: local and global solvers.

### Local solvers

Local solvers aim to produce discrete failure examples. The output of the solver is a set of likely action traces that lead to failure. The system is typically reset with a fixed initialization at the beginning of each episode. 

#### Monte Carlo search

Monte Carlo search (MCS) is a simple, uninformed local solver. It samples the environment randomly and maintains a set of best-performing action traces. It is generally not advisable to use this solver for actual stress testing, but it is useful as a baseline against other solvers, in terms of runtime performance and overall efficacy. MCS offers the following parameters:

| Parameter | Type | Default | Description |
| - | - | - | - | 
| `num_iterations` | `Int64` | `1000` | Number of episodes to test |
| `top_k` | `Int64` | `10` | Number of best-performing episodes to store |

#### Monte Carlo tree search

Monte Carlo tree search (MCTS) is a reinforcement learning algorithm that attempts to balance exploration and experience, constructing a partially-ordered search tree over actions to efficiently perform its search. It is a robust and versatile solver which can be applied to nearly any problem. MCTS offers the following parameters:

| Parameter | Type | Default | Description |
| - | - | - | - | 
| `num_iterations` | `Int64` | `1000` | Number of episodes to test |
| `top_k` | `Int64` | `10` | Number of best-performing episodes to store |
| `k` | `Float64` | `1.0` | Tree expansion coefficient |
| `α` | `Float64` | `0.7` | Tree expansion exponent |
| `c` | `Float64` | `1.0` | Exploration balance coefficient |

For an example of a problem solved with MCTS, see the notebook `examples/walk1d`.

### Global solvers

Global solvers aim to produce a mapping from the system state space to an adversarial policy. The output of the solver is a function that takes as input an observation of the system and returns an action. In this way, failure trajectories can be produced from any given initialization. This opens the door to a richer analysis of the system's weaknesses.

#### Soft actor critic

>This feature is contained in a submodule, and must be enabled for use.

Soft actor critic (SAC) is a deep reinforcement learning algorithm that simultaneously learns a value function and a policy for the `ASTMDP`. These take the form of neural networks, which can be used to generate failures or analyze system properties. SAC offers the following parameters:

| Parameter | Type | Default | Description |
| - | - | - | - | 
| `obs_dim` | `Int64` | none | Dimension of observation space | 
| `act_dim` | `Int64` | none | Dimension of action space | 
| `act_mins` | `Vector{Float64}` | none | Minimum values of actions | 
| `act_maxs` | `Vector{Float64}` | none | Maximum values of actions | 
| `gamma` | `Float64` | `0.999` | Discount factor | 
| `max_buffer_size` | `Int64` | `100000` | Maximum number of timesteps in buffer | 
| `hidden_sizes` | `Vector{Int}` | `[100,100,100]` | Dimensions of hidden layers | 
| `num_q` | `Int64` | `2` | Size of critic ensemble | 
| `activation` | `Function` | `SoftActorCritic.relu` | Activation after each hidden layer | 
| `q_optimizer` | `Any` | `AdaBelief(1e-4)` | Optimizer for value networks | 
| `pi_optimizer` | `Any` | `AdaBelief(1e-4)` | Optimizer for policy network | 
| `alpha_optimizer` | `Any` | `AdaBelief(1e-4)` | Optimizer for alpha | 
| `batch_size` | `Int64` | `64` | Size of each update to networks | 
| `epochs` | `Int64` | `200` | Number of epochs | 
| `steps_per_epoch` | `Int64` | `200` | Steps of simulation per epoch | 
| `start_steps` | `Int64` | `1000` | Steps before following policy | 
| `max_ep_len` | `Int64` | `50` | Maximum number of steps per episode | 
| `update_after` | `Int64` | `300` | Steps before networks begin to update | 
| `update_every` | `Int64` | `50` | Steps between updates | 
| `num_batches` | `Int64` | `update_every` | Number of batches per update | 
| `polyak` | `Float64` | `0.995` | Target network averaging parameter | 
| `target_entropy` | `Float64` | `-act_dim` | Target entropy (default is heuristic) | 
| `rng` | `AbstractRNG` | `Random.GLOBAL_RNG` | Random number generator | 
| `num_test_episodes` | `Int64` | `100` | Number of test episodes | 
| `displays` | `Vector{<:Tuple}` | `Tuple[]` | List of tuples containing name and callback evaluated at trajectory termination | 
| `save` | `Bool` | `false` | Enable checkpointing | 
| `save_every` | `Int64` | `10000` | Steps between checkpoints | 
| `save_dir` | `String` | `DEFAULT_SAVE_DIR` | Directory to save checkpoints | 
| `max_saved` | `Int64` | `0` | Maximum number of checkpoints; set to zero or negative for unlimited | 

For an example of a problem solved with MCTS, see the notebook `examples/walk1d`. 

## Analysis

Analysis modules provide methods to further analyze results.

### Policy-value verification

>This feature is contained in a submodule, and must be enabled for use.

Policy-value verification (PVV) is an experimental method of analyzing the output of global solvers. It assembles the policy network and value network (or ensemble of value networks) into a single value function over state space. Then, given a set condition on the value function, the algorithm uses an adaptive refinement process to classify regions of state space that provably satisfy the condition, violate the condition, or are unprovable at the given tolerance.

As a matter of ongoing research, requirements concerning the safety of the system can be linked to conditions on the value function. For instance, a requirement that the possibility of failure not exceed $10^{-9}$ from a set of initial states (given some modeled environmental stochasticity) translates to a constraint on the value function. The validity and practicality of this analysis is largely dependent on the learning process and is still uncertain. Nonetheless, the approach can currently generate *approximate* artifacts that may be useful for casual and nonrigorous analysis of system performance. 

To use PVV, a global result object is passed into the `mean_network` or `spread_network` functions to generate composite neural networks representing statistics of the learned value ensemble. A `CrossSection` or `LinearCrossSection` can be defined to reduce the dimensionality of the input space for the purposes of visualization and reduced computational burden. Finally, a `BinaryRefinery` or `IntervalRefinery` object is created to specify the condition imposed on the value function. The function `analyze` initiates the analysis algorithm. 

For an example of a problem analyzed with PVV, see the notebooks `examples/walk2d` and `examples/pvv`. 

