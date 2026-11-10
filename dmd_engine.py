import numpy as np
import matplotlib.pyplot as plt

def get_dmd_decomposition(
    data_mats: list[np.ndarray], 
    weights_mats: None|list[np.ndarray] = None, 
    n_embed = -1, 
    plot_singular_vals:bool = False,
    show_singular_vals:bool = False, 
    singular_vals_figpath:None|str = None
):
    """
    Compute the reduced DMD operator and the matrices for projection to and from its subspace.

    Parameters
    ----------
    data_mats : list of ndarray
        State sequences, each with shape (n_states, state_size).
    weights_mats : list of ndarray, optional
        Weights corresponding to each sequence, each with length n_states - 1.
    n_embed : int, optional
        Number of singular vectors to retain. A negative value retains all singular vectors.
    plot_singular_vals : bool, optional
        If True, plot the singular values and save the figure to
        ``./results/dmd_singular_values.png``.
    show_singular_vals : bool, optional
        Unused.
    singular_vals_figpath : str, optional
        Unused.

    Returns
    -------
    A_t : ndarray
        Reduced DMD operator, with shape (n_embed, n_embed).
    U_t_H : ndarray
        Conjugate transpose of the retained left singular vectors, with shape
        (n_embed, state_size).
    U_t : ndarray
        Retained left singular vectors, with shape (state_size, n_embed).
    """
    # state_size = data_mats[0].shape[1]

    # concatenating the data matrices leads to a matrix of shape (n_states * (n_experiments - 1), state_size), 
    # then we transpose this to have each column represent a state vector at a certain timestep.
    # This way, the i-th column of X contains a state x(k), and the i-th column of Xp contains the state x(k+1) of the same simulation.
    # "Boundary states", which are the last of a simulation, are removed from X
    if weights_mats is None:
        X = np.concatenate([data[:-1, :] for data in data_mats], axis=0).T 
        Xp = np.concatenate([data[1:, :] for data in data_mats], axis=0).T
    else:
        # normalize the values for more stable numerical results
        # weights_mats = [weights / weights.mean() for weights in weights_mats]

        X = np.concatenate([ data[:-1, :] * np.sqrt(weights[:, None]) for data, weights in zip(data_mats, weights_mats) ], axis=0).T      
        Xp = np.concatenate([ data[1:, :] * np.sqrt(weights[:, None]) for data, weights in zip(data_mats, weights_mats) ], axis=0).T
                

    U, S, Vh = np.linalg.svd(X, full_matrices=False)


    n_singular_vals = S.shape[0]
    n_embed = min(n_embed, n_singular_vals)

    if plot_singular_vals:
        plt.figure()
        plt.plot(np.arange(1, n_singular_vals+1), np.abs(S))
        plt.yscale("log")
        plt.xlabel("Singular value index")
        plt.ylabel("Singular values (norm)")
        plt.title(f"First 100/{n_singular_vals} Singular values of the matrix")
        plt.xlim([0,100])

        plt.savefig(singular_vals_figpath)
        if show_singular_vals:
            plt.show()
        plt.close()

    if n_embed < 0:
        n_embed = n_singular_vals
    
    # extract low rank approximation of the SVD matrices
    U_t = U[:, :n_embed]        # (65536, 8)
    S_t = S[:n_embed]           # (8,)
    V_t = Vh[:n_embed, :]       # (8, 499)

    U_t_H = np.matrix.conjugate(U_t).T      # (8, 65536)

    V = np.matrix.conjugate(V_t).T          # (499, 8)
    A_t = U_t_H @ Xp @ V @ np.diag(1 / S_t)         # (8, 65536) x (65536, 499) x (499, 8) * (8,8) = (8, 499) x (499, 8) x (8,8) = (8,8)

    return A_t, U_t_H, U_t


def run_dmd_autoregressive(initial_state, A_t, U_t_H, U_t, num_steps):
    """
    Given an initial state and the DMD matrices, this function returns a matrix of shape (num_steps, state_size) with
    all of the future states calculated in an autoregressive manner from the DMD matrices.
    """
    state_size = initial_state.shape[0]
    out_mat = np.zeros((num_steps, state_size))

    out_mat[0, :] = initial_state
    curr_idx = 0
    while curr_idx < num_steps - 1:
        curr_state = out_mat[curr_idx]

        curr_state_embedded = U_t_H @ curr_state       # (8, 65536) x (65536,) = (8,), this is the embedded x_curr
        next_state_embedded = A_t @ curr_state_embedded   # (8,8) x (8,) = (8,), this is the next state in time as predicted by the update matrix A tilde
        next_state = U_t @ next_state_embedded         # (65536, 8) x (8,) = (65536), the reconstructed next state prediction
        
        out_mat[curr_idx + 1] = next_state

        curr_idx += 1

    return out_mat



def state_projection_error(state, U_t, U_t_H):
    """ 
    Given a state vector x(k) and the projection matrices U (U_t) and U* into the SVD subspace, this function returns
    the relative projection error, given by |x(k) - UU*x(k) | / |x(k)|
    """
    state_projected = U_t_H @ state
    state_reconstructed = U_t @ state_projected

    rel_projection_err = np.linalg.norm(state - state_reconstructed) / np.linalg.norm(state)

    return rel_projection_err