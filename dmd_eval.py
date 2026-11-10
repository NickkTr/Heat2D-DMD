import numpy as np
from heat_initial_conditions import *
import matplotlib.pyplot as plt
from utils import animate_heat_simulation
import os
from utils import train_test_split, numpy_encoder
from dmd_engine import *
import json

def evaluate_dmd(
    test_sims: list[HeatSimulation], 
    A_t: np.ndarray, 
    U_t_H: np.ndarray, 
    U_t: np.ndarray, 
    verbose: bool = False,
    save_figs: bool = False,
    figs_path: str|None = None,
    generate_animations: bool = False,
    animations_path: str|None = None
):
    if save_figs and figs_path is None:
        raise ValueError("When calling evaluate_dmd with save_figs = True, a path is also expected in the parameter \"figs_path\"")

    if generate_animations and animations_path is None:
        raise ValueError("When calling evaluate_dmd with generate_animations = True, a path is also expected in the parameter \"animations_path\"")

    # generate the folders corresponding to the figures and the animations if they do not exist
    if save_figs:
        os.makedirs(figs_path, exist_ok=True)
    if generate_animations:
        os.makedirs(animations_path, exist_ok=True)
    
    results = {
        'initial_state_proj_errors': [],
        'state_abs_errors': [],
        'state_rel_errors': []
    }
    for idx, heat_sim in enumerate(test_sims):
        if verbose:
            print(f"Running tests on simulation {idx+1}/{len(test_sims)}")
        
        # Load simulation data
        T_sol = heat_sim.sim_states
        n_steps = heat_sim.t_eval.shape[0]

        # Run autoregressive predictions with DMD
        test_sim_conds = heat_sim.initial_conds
        preds = run_dmd_autoregressive(test_sim_conds.initial_temperature.flatten(), A_t, U_t_H, U_t, num_steps=n_steps)     # (num_steps, state_size)

        # Calculate errors
        abs_errs = np.linalg.norm(np.abs(T_sol - preds), axis=1)
        state_vec_norms = np.linalg.norm(T_sol, axis=1)
        rel_errs = abs_errs / state_vec_norms

        mean_abs_errs = np.mean(np.abs(T_sol - preds), axis=1)


        results['state_mean_abs_errors'] = mean_abs_errs
        results['state_rel_errors'] = rel_errs


        # extract filename of the simulation
        sim_filename = os.path.basename(heat_sim.filepath)
        sim_number = int(sim_filename[len("simulation_") : -len(".npz")])           # assuming the simulation data is always in the format "simulation_n.npz"

        # Handle figures
        if save_figs:
            fig, (rel_ax, abs_ax) = plt.subplots(1, 2, figsize=(12, 4.8))
            simulation_steps = np.arange(0, n_steps)

            rel_ax.plot(simulation_steps, rel_errs)
            rel_ax.set_xlabel("Simulation step")
            rel_ax.set_ylabel("Relative error")
            rel_ax.set_title("Relative errors ($L^2$ norm)")

            abs_ax.plot(simulation_steps, mean_abs_errs)
            abs_ax.set_xlabel("Simulation step")
            abs_ax.set_ylabel("Mean Absolute error")
            abs_ax.set_title("Mean Absolute error")

            fig.suptitle("DMD autoregressive solution errors")

            fig_name = f"sim_{sim_number}.png"
            fig_path = os.path.join(figs_path, fig_name)
            fig.savefig(fig_path)
            plt.close()

        # Handle animations
        if generate_animations:
            if verbose:
                print("\tGenerating animation from the predictions")
            
            preds = preds.reshape(-1, heat_sim.ny, heat_sim.nx)
            animation_name = f"anim_dmd_test_{sim_number}.mp4"
            animation_path = os.path.join(animations_path, animation_name)
            animate_heat_simulation(preds, heat_sim.t_eval, save_path=animation_path)       # "./anims/dmd_predicted/"

        # Calculate initial state projection error
        proj_error = state_projection_error(T_sol[0], U_t, U_t_H)
        results['initial_state_proj_errors'].append(proj_error)
        if verbose:
            print(f"\tInitial state projection error: {proj_error:.2e}\n")


    return results



def main():
    TRAIN_RATIO = 0.8
    N_EMBED = 500
    USE_WEIGHTS = False
    TAU = 40            # decay parameter in weights
    RESULTS_OUTPUT_FILE = "./results/evaluation_settings.json"          # json where the output errors and test settings will be dumped

    # Automatically extract simulation files as those whose name starts with "simulation" and whose extension is .npz
    dirpath = os.path.dirname(__file__)
    data_dir = os.path.join(dirpath, "data", "simulations")
    sim_files = [os.path.join(data_dir, x) for x in os.listdir(data_dir) if (os.path.splitext(x)[1] == ".npz" and x.startswith("simulation"))]
    num_simulations = len(sim_files)
    print(f"{num_simulations} simulation files found")

    # Extract training and testing simulations
    train_sim_names, test_sim_names = train_test_split(sim_files, TRAIN_RATIO)
    print(f"{len(train_sim_names)}/{num_simulations} simulations will be used for training, {len(test_sim_names)}/{num_simulations} will be used for testing\n")


    # === TRAINING PHASE
    print("Loading simulation data")
    heat_sims: list[HeatSimulation] = []
    sim_weights = [] if USE_WEIGHTS else None
    for sim_file in train_sim_names:
        heat_sim = load_heat_simulation(sim_file)
        heat_sims.append(heat_sim)

        # generate weights if required
        if USE_WEIGHTS:
            T_sol = heat_sim.sim_states
            n_instances = T_sol.shape[0] - 1
            weights = np.exp(-np.arange(0, n_instances, 1) / TAU)
            sim_weights.append(weights)
        

    print("Running DMD decomposition on the full data")
    sim_solutions = [heat_sim.sim_states for heat_sim in heat_sims]
    A_t, U_t_H, U_t = get_dmd_decomposition(sim_solutions, weights_mats=sim_weights, n_embed=N_EMBED, plot_singular_vals=True, singular_vals_figpath="./results/dmd_singular_values.png")

    # write the chosen training simulations to a file to make sure the results are reproducible

    # === TESTING PHASE
    test_sims = [load_heat_simulation(sim_file) for sim_file in test_sim_names]
    results = evaluate_dmd(test_sims, 
                           A_t, 
                           U_t_H, 
                           U_t, 
                           verbose=True, 
                           save_figs=True,
                           figs_path="./results/evaluation",
                           generate_animations=True,
                           animations_path="./results/animations")


    print("Mean projection error:", np.mean(results['initial_state_proj_errors']))
    print("Max projection error:", np.max(results['initial_state_proj_errors']))
    results['parameters'] = {
        'embed_dimension': N_EMBED,
        'used_weights': USE_WEIGHTS,
        'tau': TAU
    }
    results['train_sims'] = [ os.path.basename(name) for name in train_sim_names ]
    results['test_sims'] = [ os.path.basename(name) for name in test_sim_names ]

    print(f"Saving results to {RESULTS_OUTPUT_FILE}")
    with open(RESULTS_OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=4, default=numpy_encoder)





if __name__ == '__main__':
    main()