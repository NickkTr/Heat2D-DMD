import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from utils import animate_heat_simulation, wavelet_noise
from heat_sim_engine import simulate_heat

import argparse
import time

"""
Generate a number 'N_SIMULATIONS' of heat equation simulations with the given parameters, keeping the same 
diffusivity field for each, and saves the results in data/simulations.npz
"""


def embed_object_mask(alpha_grid, mask, alpha_obj, sigma=2.0):
    """
    Embeds an object defined by a binary mask into an alpha grid 
    using Gaussian-smoothed diffuse interface blending.

    Parameters:
        alpha_grid : 2D numpy array (background conductivity)
        mask       : 2D numpy array (1 for object, 0 elsewhere)
        alpha_obj  : float (conductivity of the embedded object)
        sigma      : float (smoothing width in grid points, ideally 1.5 - 3.0)
    """
    # convolve binary mask with Gaussian kernel
    # mode='wrap' preserves periodicity
    mask_smooth = gaussian_filter(mask.astype(float), sigma=sigma, mode='reflect')

    # smoothly blend alpha_grid and alpha_obj with linear interpolation
    alpha_new = (1.0 - mask_smooth) * alpha_grid + mask_smooth * alpha_obj
    return alpha_new


def generate_diffusivity_field(num_waves, freq_min, freq_max, gx, gy, sigma_pix, alpha_max = 2.0e-5):
    """ Generate alpha(x) using wavelet noise """
    alpha_0 =  alpha_max * 10

    alpha = wavelet_noise(num_waves, gx, gy, freq_min, freq_max, max_val = alpha_max / alpha_0)

    # obj_mask = ((gx > 0.3) & (gx < 0.55) & (gy > 0.25) & (gy < 0.4)).astype(float)
    # alpha = embed_object_mask(alpha, obj_mask, 0, sigma=sigma_pix)

    return alpha, alpha_0



def parse_args():
    N_SIMULATIONS_DEFAULT = 6
    N_EVALUATIONS_DEFAULT = 500

    parser = argparse.ArgumentParser()
    parser.add_argument("--nsims", help="Number of simulations to run", type=int, required=False, default=N_SIMULATIONS_DEFAULT)
    parser.add_argument("--nevals", help="Number of evenly spaced evaluations for each sim", type=int, required=False, default=N_EVALUATIONS_DEFAULT)
    args = parser.parse_args()

    n_simulations = args.nsims
    n_evaluations = args.nevals

    return n_simulations, n_evaluations
    


def main():
    N_SIMULATIONS, N_EVALUATIONS = parse_args()
    
    N_WAVES = 35          # number of waves in wavelet noise
    SIM_TIME_SECS = 1000
    print(f"Time between each discrete step: {SIM_TIME_SECS / N_EVALUATIONS:.2f}")

    # HARDCODED SIMULATION PARAMETERS
    nx, ny = 128, 128    # number of points used for discretization
    L = 1      # grid size (meters)
    dx = 1.0 / nx
    dy = 1.0 / ny
    alpha_max = 2e-5

    # Border for embedded object in diffusivity field
    sigma_phys = 0.02 * L      # physical scale of the border transition
    sigma_pix = (sigma_phys / dy, sigma_phys / dx)      # width of the border in pixels

    # frequencies for diffusivity field
    freq_min = 1 / L
    freq_max = 3 / L

    # Generate grid for
    X = (np.arange(nx) + 0.5) * dx
    Y = (np.arange(ny) + 0.5) * dy
    gx, gy = np.meshgrid(X, Y)

    alpha, alpha_0 = generate_diffusivity_field(N_WAVES, freq_min, freq_max, gx, gy, sigma_pix, alpha_max = alpha_max)

    plt.imshow(alpha, cmap='viridis')
    plt.title("thermal diffusivity map")
    plt.colorbar()
    plt.savefig("./data/diffusivity_map.png")
    plt.show()
    plt.close()


    # RECORD STATISTICS OVER THE ENTIRE STATE VECTOR FOR ALL SIMULATIONS AND EVALUATION STEPS
    min_temperatures = np.zeros((N_SIMULATIONS, N_EVALUATIONS))         
    max_temperatures = np.zeros((N_SIMULATIONS, N_EVALUATIONS))
    avg_temperatures = np.zeros((N_SIMULATIONS, N_EVALUATIONS))

    delay_total_sims = 0
    delay_total_anims = 0
    # Run all the simulations (could be parallelized but it doesn't seem necessary at this stage)
    for sim_idx in range(1, N_SIMULATIONS + 1):
        # Sample the initial temperature field
        initial_temperature = wavelet_noise(N_WAVES, gx, gy, freq_min, freq_max, min_val = 0, max_val = 1)

        # Run the simulation
        start_time = time.time_ns()
        T_sol, sol_time_real = simulate_heat(initial_temperature, alpha, L, alpha_0, sim_time_secs=SIM_TIME_SECS, n_evaluations=N_EVALUATIONS)
        end_time = time.time_ns()
        delay = (end_time - start_time) * 1e-9

        # Save the simulation results
        np.savez(f"./data/simulations/simulation_{sim_idx}", T_sol = T_sol, sol_time_real = sol_time_real, initial_temperature = initial_temperature, alpha = alpha)
        print(f"Simulation #{sim_idx} completed in {delay:.2f} seconds")

        # Record statistics 
        avg_temperatures[sim_idx-1] = np.mean(T_sol, axis=(1, 2))
        min_temperatures[sim_idx-1] = np.min(T_sol, axis=(1,2))
        max_temperatures[sim_idx-1] = np.max(T_sol, axis=(1,2))

        # Animate the results
        start_time = time.time_ns()
        animate_heat_simulation(T_sol, sol_time_real, save_path=f"./data/animations/anim_{sim_idx}.mp4")
        end_time = time.time_ns()
        delay_anim = (end_time - start_time) * 1e-9
        print(f"Animation for sim. #{sim_idx} generated in {delay_anim:.2f} seconds")

        delay_total_sims += delay
        delay_total_anims += delay_anim


    total_minutes, total_seconds = divmod(delay_total_sims, 60)
    total_minutes_anims, total_seconds_anims = divmod(delay_total_anims, 60)
    print(f"\nTotal simulation time: {int(total_minutes)} mins {total_seconds:.2f} seconds")
    print(f"Total time to generate the animations: {int(total_minutes_anims)} mins {total_seconds_anims:.2f} seconds")



    # === CHECK VALIDITY OF THE SOLUTIONS
    # (mean temperature should be approximately constant, and temperature should regress towards the mean over time)

    avg_fig, avg_axes = plt.subplots(
        N_SIMULATIONS, 1, figsize=(10, 3 * N_SIMULATIONS), squeeze=False
    )
    avg_fig.suptitle("Average temperature over time")
    for sim_idx in range(N_SIMULATIONS):
        ax = avg_axes[sim_idx, 0]
        ax.plot(sol_time_real, avg_temperatures[sim_idx])
        ax.set_title(f"Simulation #{sim_idx + 1}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Average temperature")
        ax.set_ylim([0, 1])

    range_fig, range_axes = plt.subplots(
        N_SIMULATIONS, 1, figsize=(10, 3 * N_SIMULATIONS), squeeze=False
    )
    range_fig.suptitle("Minimum and maximum temperatures over time")
    for sim_idx in range(N_SIMULATIONS):
        ax = range_axes[sim_idx, 0]
        ax.plot(sol_time_real, min_temperatures[sim_idx], label="Min. temperature")
        ax.plot(sol_time_real, max_temperatures[sim_idx], label="Max. temperature")
        ax.set_title(f"Simulation #{sim_idx + 1}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Temperature")
        ax.legend()

    avg_fig.tight_layout(rect=(0, 0, 1, 0.97))
    range_fig.tight_layout(rect=(0, 0, 1, 0.97))
    avg_fig.savefig("./data/avg_temps.png", bbox_inches="tight")
    range_fig.savefig("./data/temps_diffusion.png", bbox_inches="tight")






if __name__ == '__main__':
    main()
