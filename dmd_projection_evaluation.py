import numpy as np
from heat_initial_conditions import *
import matplotlib.pyplot as plt
import os
from utils import train_test_split
from dmd_engine import *








# Automatically extract simulation files as those whose name starts with "simulation" and whose extension is .npz
dirpath = os.path.dirname(__file__)
data_dir = os.path.join(dirpath, "data")
sim_files = [os.path.join(data_dir, x) for x in os.listdir(data_dir) if (os.path.splitext(x)[1] == ".npz" and x.startswith("simulation"))]
num_simulations = len(sim_files)
print(f"{num_simulations} simulation files found")

# Extract training and testing simulations
TRAIN_RATIO = 0.8
train_sims, test_sims = train_test_split(sim_files, TRAIN_RATIO)
print(f"{len(train_sims)}/{num_simulations} simulations will be used for training, {len(test_sims)}/{num_simulations} will be used for testing\n")


# === TRAINING PHASE
print("Loading simulation data")
sim_solutions = []
sim_weights = []
for sim_file in train_sims:
    heat_sim = load_heat_simulation(sim_file)
    T_sol = heat_sim.sim_states
    
    sim_solutions.append(T_sol)

    # generate weights
    n_instances = T_sol.shape[0]
    t0 = 40
    weights = np.exp(-np.arange(0, n_instances-1, 1) / t0)
    sim_weights.append(weights)
    



# for each of these embedding dimensions, check the reconstruction error both in the weighted and unweighted case 
use_weights = False
tested_embed_dimensions = [60, 80, 100, 200, 300]
mean_rel_errors = {'with_weights': [], 'without_weights': []}
max_proj_errors = {'with_weights': [], 'without_weights': []}

for use_weights in [True, False]:
    print(f"\nRunning tests for use_weights = {use_weights}")
    for dim_embed in tested_embed_dimensions:
        print(f"\tRunning DMD for dim embed = {dim_embed}")
        weights = sim_weights if use_weights else None
        A_t, U_t_H, U_t = get_dmd_decomposition(sim_solutions, weights_mats=weights, n_embed=dim_embed, plot_singular_vals=False)
        

        initial_state_proj_errors = []
        for idx, sim_file in enumerate(test_sims):
            print(f"\t\tRunning tests on simulation {idx+1}/{len(test_sims)}")
            # Extract the simulation number from the filename
            sim_filename = os.path.basename(sim_file)
            sim_number = int(sim_filename[len("simulation_") : -len(".npz")])           # assuming the simulation data is always in the format "simulation_n.npz"

            # Load simulation data
            heat_sim = load_heat_simulation(sim_file)
            T_sol = heat_sim.sim_states
            n_steps = heat_sim.t_eval.shape[0]

            proj_error = state_projection_error(T_sol[0], U_t, U_t_H)
            initial_state_proj_errors.append(proj_error)
            # print(f"Initial state projection error: {proj_error:.2e}")


        key = 'with_weights' if use_weights else 'without_weights'
        mean_rel_errors[key].append(np.mean(initial_state_proj_errors))
        max_proj_errors[key].append(np.max(initial_state_proj_errors))


fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
series = [
    ("with_weights", "Weighted DMD", "#0072B2", "o"),
    ("without_weights", "Unweighted DMD", "#D55E00", "s"),
]

for key, label, color, marker in series:
    axes[0].plot(
        tested_embed_dimensions,
        mean_rel_errors[key],
        label=label,
        color=color,
        marker=marker,
        linewidth=2,
        markersize=7,
    )
    axes[1].plot(
        tested_embed_dimensions,
        max_proj_errors[key],
        label=label,
        color=color,
        marker=marker,
        linewidth=2,
        markersize=7,
    )

axes[0].set_title("Mean error")
axes[0].set_ylabel("Mean relative projection error")
axes[1].set_title("Maximum error")
axes[1].set_ylabel("Maximum relative projection error")

for ax in axes:
    ax.set_xlabel("Embedding dimension")
    ax.set_xticks(tested_embed_dimensions)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)

fig.suptitle("Initial-state projection error by DMD embedding dimension (using $L^2$ norms over state vectors)", fontsize=14)
fig.tight_layout()

results_dir = os.path.join(dirpath, "results")
os.makedirs(results_dir, exist_ok=True)
fig.savefig(
    os.path.join(results_dir, "dmd_projection_errors.png"),
    dpi=200,
    bbox_inches="tight",
)
plt.show()
