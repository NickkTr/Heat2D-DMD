import numpy as np
import matplotlib.pyplot as plt
import datetime
from matplotlib.animation import FuncAnimation
from random import shuffle


def animate_heat_simulation(T_series, t_vec=None, interval=50, save_path=None, cmap='inferno'):
    """
    Animates a 2D heat simulation time-series on a heatmap.

    Parameters:
        T_series  : 3D numpy array of shape (num_frames, ny, nx)
        t_vec     : 1D numpy array or list of time values (e.g., sol.t)
        interval  : int, delay between frames in milliseconds
        save_path : str (optional), file path to export animation (e.g. 'heat_sim.gif' or 'heat_sim.mp4')
        cmap      : str, Matplotlib colormap ('inferno', 'plasma', 'hot', 'viridis')
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    # Lock color limits across all time steps to prevent fluctuating colorbar scales
    vmin = np.min(T_series)
    vmax = np.max(T_series)

    # Initialize heatmap display with frame 0
    im = ax.imshow(T_series[0], cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, label='Temperature')

    initial_time_str = f"{t_vec[0]:.3f}" if t_vec is not None else "0"
    title = ax.set_title(f"Time: {initial_time_str} | Frame 0/{len(T_series)-1}")
    ax.set_xlabel("x grid")
    ax.set_ylabel("y grid")

    def update(frame):
        im.set_array(T_series[frame])
        if t_vec is not None:
            time_str = str(datetime.timedelta(seconds=t_vec[frame]))
            dot_idx = time_str.find(".")
            if dot_idx != -1:
                time_str = time_str[:dot_idx]

            time_text = f"Time: {time_str}"
        else:
            time_text = f"Frame {frame}"

        title.set_text(f"{time_text} | Frame {frame}/{len(T_series)-1}")
        return im, title

    anim = FuncAnimation(
        fig, 
        update, 
        frames=len(T_series), 
        interval=interval, 
        blit=False
    )

    if save_path:
        fps = max(1, int(1000 / interval))
        if save_path.endswith('.gif'):
            anim.save(save_path, writer='pillow', fps=fps)
        elif save_path.endswith('.mp4'):
            anim.save(save_path, writer='ffmpeg', fps=fps)

    plt.close(fig)
    return anim



def wavelet_noise(num_waves, gx, gy, freq_min, freq_max, min_val = 0, max_val=2.0e-7):
    """ Generates a 2d grid of wavelet noise by sampling random frequences between freq_min and freq_max """
    alpha = np.zeros(gx.shape)

    for i in range(num_waves + 1):
        # generate the wave frequency and cosine/sine coefficients
        freq = np.random.uniform(low=freq_min, high=freq_max)
        w = 2 * np.pi * freq
        coeffs = np.random.randn(2) * 0.2

        # sample the wave direction
        wave_dir = np.random.randn(2)
        wave_dir = wave_dir / np.linalg.norm(wave_dir)
        wave_dir_term = (wave_dir[0] * gx + wave_dir[1] * gy)

        # calculate the actual wave
        wave_term = coeffs[0] * np.cos(w * wave_dir_term) + coeffs[1] * np.sin(w * wave_dir_term)
        alpha += wave_term

    alpha = alpha - np.min(alpha)                # make sure alpha is non-negative everywhere
    alpha = min_val + alpha * (max_val - min_val) / np.max(alpha)       # make sure the highest value is max_val 

    return alpha




def train_test_split(data: list, train_ratio: float):
    """ 
    Given a list of simulation data with N simulations (ex: numpy arrays, paths to files with the data etc.), this function randomly partitions the original 
    list into two sublists data_train and data_test, where data_train has the data for a fraction train_ratio of the original simulations, and 
    data_test has the rest.
    """
    num_simulations = len(data)
    num_train_simulations = int(np.floor(num_simulations * train_ratio)) 

    data_copy = data.copy()
    shuffle(data_copy)

    data_train = data_copy[:num_train_simulations]
    data_test = data_copy[num_train_simulations:]

    return data_train, data_test



def numpy_encoder(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")