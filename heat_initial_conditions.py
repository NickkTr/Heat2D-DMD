import numpy as np
from dataclasses import dataclass


@dataclass
class HeatSimulationConditions:
    initial_temperature: np.ndarray         # initial temperature field    
    alpha: np.ndarray                       # thermal diffusivity field




@dataclass
class HeatSimulation:
    initial_conds: HeatSimulationConditions
    sim_states: np.ndarray                  # matrix of shape (n_steps, state_size) with all the states in the simulation
    t_eval: np.ndarray                      # time stamps where the solution is to be evaluated (seconds)
    nx: int                                 # grid width
    ny: int                                 # grid height
    filepath:None|str                       # the filepath this simulation was originally loaded from (if present)



def load_heat_simulation(path) -> HeatSimulation:
    """ 
    Loads the results of a heat simulation from a given path. Returns the simulation conditions and 
    the computed solution in the shape (n_evaluations, ny*nx) 
    """
    npzfile = np.load(path)
    T_sol = npzfile["T_sol"]
    sol_time_real = npzfile["sol_time_real"]
    initial_temperature = npzfile["initial_temperature"]
    alpha = npzfile["alpha"]

    n_evaluations, ny, nx = T_sol.shape

    T_sol = T_sol.reshape(n_evaluations, ny*nx)

    initial_conds = HeatSimulationConditions(initial_temperature, alpha)
    heat_sim = HeatSimulation(initial_conds, T_sol, sol_time_real, nx, ny, path)
    
    return heat_sim