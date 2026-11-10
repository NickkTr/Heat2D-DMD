import numpy as np
from scipy.integrate import solve_ivp



def calculate_x_gradient(f, dx):
    """ Given a scalar field f represented as a matrix (ny, nx), this function computes and returns a matrix of the same shape with elements given by ∂f/∂x """
    df_dx = np.zeros_like(f)
    # centered 1st derivative for interior: (f[i+1] - f[i-1]) / (2*dx)
    df_dx[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2 * dx)
    
    # 2nd order one sided differences at boundaries to preserve O(dx²) accuracy
    df_dx[:, 0]  = (-3 * f[:, 0] + 4 * f[:, 1] - f[:, 2]) / (2 * dx)
    df_dx[:, -1] = (3 * f[:, -1] - 4 * f[:, -2] + f[:, -3]) / (2 * dx)

    return df_dx

def calculate_y_gradient(f, dy):
    """ Given a scalar field f represented as a matrix (ny, nx), this function computes and returns a matrix of the same shape with elements given by ∂f/∂y """
    df_dy = np.zeros_like(f)
    # centered 1st derivative for interior: (f[i+1] - f[i-1]) / (2*dy)
    df_dy[1:-1, :] = (f[2:, :] - f[:-2, :]) / (2 * dy)

    # 2nd order one sided differences at boundaries to preserve O(dy²) accuracy
    df_dy[0, :]  = (-3 * f[0, :] + 4 * f[1, :] - f[2, :]) / (2 * dy)
    df_dy[-1, :] = (3 * f[-1, :] - 4 * f[-2, :] + f[-3, :]) / (2 * dy)

    return df_dy

def calculate_gradients(f, dx, dy):
    """Computes first spatial derivatives (df/dx, df/dy) using finite differences."""
    df_dx = calculate_x_gradient(f, dx)
    df_dy = calculate_y_gradient(f, dy)

    return df_dx, df_dy


def apply_neumann_condition(df_dx, df_dy):
    """ 
    Given two matrices with shape (ny, nx) representing the derivatives ∂f/∂x and ∂f/∂y of a scalar field on a grid, this function applies the 
    Neumann condition ∇T · ̂n = 0 on the boundary ∂Ω, where ̂n is the vector normal to the boundary.
    """
    df_dx[:, 0] = 0
    df_dx[:, -1] = 0

    df_dy[0, :] = 0
    df_dy[-1, :] = 0

    return df_dx, df_dy


def compute_rhs(T_mat, alpha, dx, dy):
    """ Computes ∂T/∂t for the heat equation """
    dT_dx, dT_dy = calculate_gradients(T_mat, dx, dy)
    dT_dx, dT_dy = apply_neumann_condition(dT_dx, dT_dy)

    tmp1 = calculate_x_gradient(alpha * dT_dx, dx)
    tmp2 = calculate_y_gradient(alpha * dT_dy, dy)

    dT_dt = tmp1 + tmp2


    return dT_dt


def scipy_wrapper(t, T_flat, alpha, dx, dy, ny, nx):
    T_mat = T_flat.reshape((ny, nx))
    rhs = compute_rhs(T_mat, alpha, dx, dy)
    return rhs.ravel()


def simulate_heat(initial_temperature, alpha, L, alpha_0, sim_time_secs = 1000, n_evaluations=500):
    t0 = L**2 / alpha_0
    t_final = sim_time_secs / t0

    ny, nx = initial_temperature.shape

    dx = 1.0 / nx
    dy = 1.0 / ny

    sol = solve_ivp(
        fun = scipy_wrapper,
        t_span = (0.0, t_final),
        y0 = initial_temperature.ravel(),
        method='RK45',  
        t_eval = np.linspace(0.0, t_final, n_evaluations),
        args = (alpha, dx, dy, ny, nx),
        rtol=1e-6, atol=1e-9,
    )
    T_sol = sol.y.T.reshape(-1, ny, nx)
    sol_time_real = sol.t * t0               # array of time stamps in which the solution is evaluated in seconds
    return T_sol, sol_time_real