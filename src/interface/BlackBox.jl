# To be implemented by user.

"""
Parent type for user simulation in which neither state nor environment is exposed.
"""
abstract type BlackBox <: AbstractSimulation end


"""
Resets simulation.
"""
function reset!(sim::BlackBox)::Nothing end


"""
Steps simulation and returns log probability of environment.
"""
function step!(sim::BlackBox)::Real end


"""
Checks whether simulation has finished due to time limit or terminal state, independent of event status.
"""
function isterminal(sim::BlackBox)::Bool end


"""
Checks whether simulation is in an event state.
"""
function isevent(sim::BlackBox)::Bool end


"""
Returns custom metric of distance to event. For best results, metric should depend only on current state.
"""
function distance(sim::BlackBox)::Real end
