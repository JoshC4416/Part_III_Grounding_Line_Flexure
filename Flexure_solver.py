import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.sparse import lil_matrix
import pandas as pd
import math
np.math = math


def Derivative_matrix(x, q, n_diff):
    
    # Stencil parameter - (2q+1) stencil, so I need 2q>=n_diff
    if 2 * q < n_diff:
        raise ValueError(f"Stencil parameter must be >= {int(np.ceil(n_diff / 2))} for this order of derivative!")

    # Spatial grid parameters
    N = len(x)
    dx = x[-1] - x[-2]

    
    # Function to calculate finite difference coefficients
    # def finite_differences_coeffs(v, n_diff):
    #     M = len(v)
    #     A = np.zeros((M, M))
    #     b = np.zeros(M)
    #     b[n_diff] = np.math.factorial(n_diff)
    #     for i in range(M):
    #         A[i, :] = v**i / np.math.factorial(i)

    #     return np.linalg.solve(A, b)
    def finite_differences_coeffs(v, n_diff):
        """Returns finite difference coefficients for the given stencil."""
        M = len(v)
        A = np.zeros((M, M))
        b = np.zeros(M)
        b[n_diff] = np.math.factorial(n_diff)
        for i in range(M):
            A[i, :] = v**i / np.math.factorial(i)
        return np.linalg.solve(A, b)
    

    # def finite_differences_coeffs(reference_points, n_derivative):
    #     reference_points = np.array(reference_points, dtype=float).flatten()
    #     N = len(reference_points)

    #     if n_derivative >= N:
    #         raise ValueError(f"For this number of reference points, you can only determine derivatives of order from 0 to {N - 1}.")

    #     M = np.ones((N, N))
    #     derivative_vector = np.zeros(N)
    #     derivative_vector[n_derivative] = 1

    #     non_zero_points = reference_points[reference_points != 0]
    #     if non_zero_points.size == 0:
    #         dref_min = 1.0
    #     else:
    #         dref_min = np.min(np.abs(non_zero_points))

    #     reference_points = reference_points / dref_min

    #     for count in range(1, N):
    #         M[count, :] = M[count - 1, :] * reference_points / count

    #     weightings = np.linalg.solve(M, derivative_vector)
    #     weightings = (dref_min ** (-n_derivative)) * weightings

    #     return weightings

    # Preallocate sparse matrix
    # nnz_elements = N + q * (2 * N - 1) - q**2
    D_mat = lil_matrix((N, N), dtype=np.float64)
    # D_mat = np.zeros((N, N))

    # Build the bulk of the matrices
    # These are the points where we can look q to the left and q to the right and not look past the edges of the domain
    for m in range(q, N - q):
        v = x[m - q:m + q + 1] - x[m]
        D_mat[m, m - q:m + q + 1] = finite_differences_coeffs(v, n_diff)


    # Fill in the first q rows (Look Right)
    for m in range(q):
        v = x[:2 * q + 1] - x[m]
        D_mat[m, :2 * q + 1] = finite_differences_coeffs(v, n_diff)

    # Fill in the final q rows (Look Left)
    for m in range(N - q, N):
        v = x[-(2 * q + 1):] - x[m]
        D_mat[m, -2 * q - 1:] = finite_differences_coeffs(v, n_diff)

    return D_mat



def contiguous_regions(v):
    N = len(v)
    d = np.diff(np.sign(v.astype(int)))  # Compute the difference in the sign of the array

    idxs_start = np.where(d > 0)[0] + 1  # Indices where regions start
    if v[0] > 0:
        idxs_start = np.insert(idxs_start, 0, 0)  # Include start if first value is > 0

    idxs_end = np.where(d < 0)[0]  # Indices where regions end
    if v[-1] > 0:
        idxs_end = np.append(idxs_end, N - 1)  # Include end if last value is > 0

    # Combine start and end indices into a list of tuples
    idxs = list(zip(idxs_start, idxs_end))

    return idxs


def force_balance(zm, h, b0, x, C1, C2, C3, plot_flag):

    def plot_force_balance(x, zm, h, b0, Residual, Bending, Weight, Floating, Grounded):
        """
        Plot the force balance terms.
        """
        Nx = len(x)
        f = zm - h / 2
        v_gr = f <= -b0

        # Identify contiguous grounded regions
        idxs_gr = contiguous_regions(v_gr)

        # fig, axs = plt.subplots(7, 1, figsize=(10, 8), sharex=True)
        fig, axs = plt.subplot_mosaic([[0],[0],[2],[3],[4],[5],[6]], figsize=(10, 8), sharex=True, constrained_layout=True)
        fig.suptitle('Force Balance', fontsize=20)

        # Plot Elastic Layer and Bathymetry
        axs[0].plot(x, -b0, 'r-', label='Bathymetry')
        axs[0].plot(x, zm, 'b-', label='Elastic Layer')
        axs[0].set_ylabel('$z$ (m)', fontsize=12)
        axs[0].fill_between(x, zm - h / 2, zm + h / 2, color='blue', alpha=0.1, label='Grounded region')
        axs[0].legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))
        axs[0].grid()

        # Plot individual terms
        for i, (term, color, title) in enumerate(zip(
                [Bending, Weight, Floating, Grounded, Residual],
                ['blue', 'red', 'magenta', 'green', 'black'],
                ['Bending', 'Weight', 'Flotation', 'Grounded', 'Residual'])):
            axs[i + 2].plot(x, term, color=color, label=title)
            axs[i + 2].fill_between(x, term, color=color, alpha=0.1)
            axs[i + 2].set_title(title)
            axs[i + 2].grid()

        axs[6].set_ylim(-10, 10)

        plt.xlabel('x (m)')
        plt.tight_layout()
        plt.show()

        return

    Nx = len(x)

    # Derivative matrices (assuming you have equivalent Python implementations)
    q_diff = 4
    D1_mat = Derivative_matrix(x, q_diff, 1)
    D2_mat = Derivative_matrix(x, q_diff, 2)
    D3_mat = Derivative_matrix(x, q_diff, 3)
    D4_mat = Derivative_matrix(x, q_diff, 4)

    # Bending Term
    d0h3_mat = diags(h**3)
    d1h3_mat = diags(D1_mat @ h**3)
    d2h3_mat = diags(D2_mat @ h**3)

    M_bend = C1 * (d2h3_mat @ D2_mat + 2 * d1h3_mat @ D3_mat + d0h3_mat @ D4_mat)
    Bending = M_bend @ zm

    # Weight Term
    Weight = C3 * h

    # Floating and Grounded Terms
    f = h / 2 - zm
    b0 = b0.flatten()

    v_gr = (-f <= -b0)
    v_ungr = (-f > -b0)
    v_sb = (-f <= 0)
    v_flt = v_sb  # Floating region (submerged)

    Grounded = np.zeros(Nx)
    Grounded[v_gr] = C2 * (f[v_gr] - b0[v_gr])

    Floating = np.zeros(Nx)
    Floating[v_flt] = f[v_flt]

    # Residual
    Residual = Bending + Weight - (Floating + Grounded)

    # Plot if flag is set
    if plot_flag:
        plot_force_balance(x, zm, h, b0, Residual, Bending, Weight, Floating, Grounded)

    return Residual, Bending, Weight, Floating, Grounded



def zm_solver(zm_0, h, b0, M_bend, C2, C3):

    Nx = len(h)

    # Indicator profiles for grounded and submerged regions
    grounded_region = ((zm_0 - (h/2)) <= -b0).astype(int)                              # Grounded
    # submerged_region_grounded_and_ungrounded = ( ((zm_0 - h / 2) <= 0)
    submerged_region = ((zm_0 - h / 2) <= 0).astype(int) # & ( (zm_0 - h / 2) > -b0))  # Submerged & ungrounded

    # To ignore flotation when the ice compresses the bed below the sea surface, replace H0 by ((1-Hb)* H0)
    M = M_bend + C2 * diags(grounded_region, offsets=0, shape=(Nx, Nx)) + diags(submerged_region, offsets=0, shape=(Nx, Nx))
    RHS = (C2 * ((h/2) - b0) * grounded_region + ((h/2) - 0) * submerged_region - C3 * h)

    # Apply BCs
    M = M.toarray()
    # Flotation BC at the left, grounded BC at the right

    # zm(0) = (1/2 - rho_i/rho_w)*h(0)
    M[0, :] = 0
    M[0, 0] = 1
    RHS[0] = (1/2 - C3) * h[0]

    # zm(1) = (1/2 - rho_i/rho_w)*h(1)
    M[1, :] = 0
    M[1, 1] = 1
    RHS[1] = (1/2 - C3) * h[1]

    ############
    for i in range(2, 10):
        M[i, :] = 0
        M[i, i] = 1
        RHS[i] = (1/2 - C3) * h[i]
    ############

    # Compression without buoyancy
    M[-2, :] = 0
    M[-2, -2] = 1
    RHS[-2] = -b0[-2] + (1/2 - C3/C2) * h[-2]

    M[-1, :] = 0
    M[-1, -1] = 1
    RHS[-1] = -b0[-1] + (1/2 - C3/C2) * h[-1]

    ############
    for i in range(3, 11):
        M[-i, :] = 0
        M[-i, -i] = 1
        RHS[-i] = -b0[-i] + (1/2 - C3/C2) * h[-i]
    ############

    # zm = spsolve(M.tocsr(), RHS)
    # zm = spsolve(M, RHS)
    zm = np.linalg.solve(M, RHS)

    return(zm)



def dzm_dh_solver(zm, h, b0, M_bend, M_bend_dh, C2, C3):

    Nx = len(h)

    # Indicator profiles for grounded and submerged regions
    grounded_region = ((zm - (h/2)) <= -b0).astype(int)                             # Grounded
    # submerged_region_grounded_and_ungrounded = ( ((zm_0 - h / 2) <= 0)
    submerged_region = ((zm - h/2) <= 0).astype(int) # & ((zm - h/2) > -b0)).astype(int)         # Submerged & ungrounded
    
    # Build the matrix and RHS for the discretised ODE
    M = M_bend + C2 * diags(grounded_region, offsets=0, shape=(Nx, Nx)) + diags(submerged_region, offsets=0, shape=(Nx, Nx))
    
    RHS = ((-M_bend_dh @ zm) + (1/2 * (C2*grounded_region + submerged_region)) - C3)

    # Apply BCs
    M = M.toarray()
    # Flotation BC at the left, grounded BC at the right

    # zm(1) = ( 1/2 - rho_i/rho_w )*h(1)
    M[0, :] = 0
    M[0, 0] = 1
    RHS[0] = (1/2 - C3)

    # zm(2) = ( 1/2 - rho_i/rho_w )*h(2)
    M[1, :] = 0
    M[1, 1] = 1
    RHS[1] = (1/2 - C3)

    ############
    for i in range(2, 10):
        M[i, :] = 0
        M[i, i] = 1
        RHS[i] = (1/2 - C3)
    ############

    # Compression without buoyancy
    M[-2, :] = 0
    M[-2, -2] = 1
    RHS[-2] = (1/2 - C3/C2)

    M[-1, :] = 0
    M[-1, -1] = 1
    RHS[-1] = (1/2 - C3/C2)

    ############
    for i in range(3, 11):
        M[-i, :] = 0
        M[-i, -i] = 1
        RHS[-i] = (1/2 - C3/C2)
    ############

    # dzm_dh = spsolve(M, RHS)
    dzm_dh = np.linalg.solve(M, RHS)
    # print(dzm_dh)

    return(dzm_dh)



def dzm_db0_solver(zm, h, b0, M_bend, C2):

    Nx = len(h)

    # Indicator profiles for grounded and submerged regions
    grounded_region = ((zm - (h/2)) <= -b0).astype(int)                              # Grounded
    # submerged_region_grounded_and_ungrounded = (((zm_0 - h/2) <= 0)
    submerged_region = ((zm - h/2) <= 0).astype(int) # & ((zm - h/2) > -b0)).astype(int)  # Submerged & ungrounded

    # Build the matrix and RHS for the discretised ODE
    M = M_bend + C2 * diags(grounded_region, offsets=0, shape=(Nx, Nx)) + diags(submerged_region, offsets=0, shape=(Nx, Nx))
    
    RHS = (-C2) * grounded_region

    # Apply BCs
    M = M.toarray()
    # Flotation BC at the left, grounded BC at the right

    # # Convert matrix M to a list of lists to be able to call indices
    # M = M.tolil()

    # zm(1) = ( 1/2 - rho_i/rho_w )*h(1)
    M[0, :] = 0
    M[0, 0] = 1
    RHS[0] = 0

    # zm(2) = ( 1/2 - rho_i/rho_w )*h(2)
    M[1, :] = 0
    M[1, 1] = 1
    RHS[1] = 0

    ############
    for i in range(2, 10):
        M[i, :] = 0
        M[i, i] = 1
        RHS[i] = 0
    ############


    # Compression without buoyancy
    M[-2, :] = 0
    M[-2, -2] = 1
    RHS[-2] = -1

    M[-1, :] = 0
    M[-1, -1] = 1
    RHS[-1] = -1

    ############
    for i in range(3, 11):
        M[-i, :] = 0
        M[-i, -i] = 1
        RHS[-i] = -1
    ############

    # dzm_db0 = spsolve(M, RHS)
    dzm_db0 = np.linalg.solve(M, RHS)

    # print(dzm_db0) # Used for debugging

    return(dzm_db0)



def Flexure_profile_unfixed_grounding_line(zm_0, h, b0, x, C1, C2, C3, omega, quiet, force_balance_flag):

    # This function iterates to find the centre-line profile zm for a given uncompressed bed topography b0(x)
    # and ice thickness h(x), along with an initial guess zm_0 and physical parameter constants:

    #              C1 = gamma_val / (rho_w * g)    [m]
    #              C2 = k0 / (rho_w * g)           [-]
    #              C3 = rho_i / rho_w              [-]

    # The grounded and ungrounded regions are not imposed here, but instead determined as the solution evolves

    Nx = len(x)

    # Build matrices that doesnt change between iterations

    q_diff = 4

    D1_mat = Derivative_matrix(x, q_diff, 1)
    D2_mat = Derivative_matrix(x, q_diff, 2)
    D3_mat = Derivative_matrix(x, q_diff, 3)
    D4_mat = Derivative_matrix(x, q_diff, 4)

    d0h3_mat = diags(h**3, offsets=0, shape=(Nx, Nx))
    d1h3_mat = diags(D1_mat @ h**3, offsets=0, shape=(Nx, Nx))
    d2h3_mat = diags(D2_mat @ h**3, offsets=0, shape=(Nx, Nx))

    # M_bend = C1 * D2_mat * spdiags(h**3, offsets=0, shape=(Nx, Nx)) * D2_mat
    M_bend = C1 * ((d2h3_mat @ D2_mat) + 2 * (d1h3_mat @ D3_mat) + (d0h3_mat @ D4_mat))
    
    # print(f"Condition number: {np.linalg.cond(M_bend.toarray())}")
    
    # Version for the sensitivity calculation
    d0h2_mat = diags(h**2, offsets=0, shape=(Nx, Nx))
    d1h2_mat = diags((D1_mat @ h**2), offsets=0, shape=(Nx, Nx))
    d2h2_mat = diags((D1_mat @ h**2), offsets=0, shape=(Nx, Nx))

    M_bend_dh = C1 * 3 * ((d2h2_mat @ D2_mat) + 2 * (d1h2_mat @ D3_mat) + (d0h2_mat @ D4_mat))

    ## Centreline profile

    # Parameters for the iteration process

    error_threshold = 1e-6
    error_val = 10 * error_threshold

    # Relaxation parameter
    # omega = 0.75

    # Iterative process
    run_count = 0

    while error_val > error_threshold:

        zm = zm_solver(zm_0, h, b0, M_bend, C2, C3)

        # Tempered relaxation
        zm = zm_0 + omega * (zm - zm_0)

        # Quantify the change in the profile
        # L2 norm between new and old solutions
        error_val = (1 / Nx) * np.linalg.norm(zm_0 - zm, ord=2)
        # maximum difference between new and old solutions
        # error_val = max(abs(zm_0 - zm))

        # Update the 'old' profile for the next iteration
        zm_0 = zm.copy()

        run_count += 1

    Residual, _, _, _, _ = force_balance(zm, h, b0, x, C1, C2, C3, force_balance_flag)

    # print(run_count)
    # Sensitivity profiles
    dzm_dh  = dzm_dh_solver(zm, h, b0, M_bend, M_bend_dh, C2, C3)

    dzm_db0 = dzm_db0_solver(zm, h, b0, M_bend, C2)

    if (~quiet):
        print(" - Calculated zm")
        print(f"    - run_count = {run_count}")
        print(f"    - Force Residual = {np.linalg.norm(Residual[2:Nx-2], 2) / (Nx - 4):.3e}")

    # Sensitivity profiles
    dzm_dh = dzm_dh_solver(zm, h, b0, M_bend, M_bend_dh, C2, C3)
    if (~quiet):
        print(" - Calculated dzm_dh")

    dzm_db0 = dzm_db0_solver(zm, h, b0, M_bend, C2)
    if (~quiet):
        print(" - Calculated dzm_db0")

    # return midline flexure profile
    return(zm, dzm_dh, dzm_db0)