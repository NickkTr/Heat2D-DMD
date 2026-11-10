import numpy as np
from heat_initial_conditions import *
import os
import random
from dmd_eval import evaluate_dmd
from dmd_engine import *
from utils import numpy_encoder
import json

dirpath = os.path.dirname(__file__)
data_dir = os.path.join(dirpath, "data", "simulations")
sim_files = [os.path.join(data_dir, x) for x in os.listdir(data_dir) if (os.path.splitext(x)[1] == ".npz" and x.startswith("simulation"))]
num_simulations = len(sim_files)
print(f"{num_simulations} simulation files found")
TRAIN_RATIO = 0.8


# LOAD ALL THE AVAILABLE SIMULATIONS FIRST
print("Loading simulation data")
heat_sims: list[HeatSimulation] = []
sim_weights = None
for sim_file in sim_files:
    heat_sim = load_heat_simulation(sim_file)
    heat_sims.append(heat_sim)

    # generate weights
    # T_sol = heat_sim.sim_states
    # n_instances = T_sol.shape[0]
    # t0 = 40
    # weights = np.exp(-np.arange(0, n_instances-1, 1) / t0)
    # sim_weights.append(weights)


all_results = {}
for num_used in [10, 20, 30, 40, 50]:
    print(f"Testing performance with {num_used} simulations")
    used_sims_indices = random.sample(range(num_simulations), num_used)
    
    num_train = int(np.floor(TRAIN_RATIO * num_used))

    # since used_sims_indices is randomly shuffled we can just take the first n_test indices for the training set
    train_sims_indices = used_sims_indices[:num_train]
    test_sims_indices = used_sims_indices[num_train:]

    selected_sims = [heat_sims[idx] for idx in train_sims_indices]
    selected_weights = None if sim_weights is None else [sim_weights[idx] for idx in train_sims_indices]



    # run DMD decomposition and save the singular values plot
    sim_solutions = [heat_sim.sim_states for heat_sim in selected_sims]
    A_t, U_t_H, U_t = get_dmd_decomposition(sim_solutions, 
                                            weights_mats=selected_weights, 
                                            n_embed=500,
                                            plot_singular_vals=True, 
                                            singular_vals_figpath=f"./results/error_vs_num_trajectories/singular_vals_{num_used}_trajectories.png")


    # Verify effectiveness on the test data
    test_sims = [heat_sims[idx] for idx in test_sims_indices]

    results = evaluate_dmd(
        test_sims,
        A_t,
        U_t_H,
        U_t,
        verbose=False,
        save_figs=True,
        figs_path=f"./results/error_vs_num_trajectories/{num_used}_trajectories"
    )
    all_results[num_used] = results


RESULTS_OUTPUT_FILE = "./results/error_vs_num_trajectories/all_results.json"
print(f"Saving results to {RESULTS_OUTPUT_FILE}")
with open(RESULTS_OUTPUT_FILE, 'w') as f:
    json.dump(all_results, f, indent=4, default=numpy_encoder)

