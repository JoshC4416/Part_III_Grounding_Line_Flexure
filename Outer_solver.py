
def Outer_profile_solver(distance, surface, bed, float_base, interp_base, velocity, idx_start, fig_freq, quiet, force_balance_flag):

    import numpy as np
    # from scipy.sparse import diags
    # from scipy.sparse.linalg import spsolve
    # from scipy.sparse import lil_matrix
    import pandas as pd
    import math
    np.math = math
    import matplotlib.pyplot as plt
    from scipy.integrate import cumulative_trapezoid

    from Flexure_solver import Flexure_profile_unfixed_grounding_line
    from Polyfit_data_gradient import polyfit_data_gradient

    import matplotlib.pyplot as plt
    # from scipy.optimize import newton

    # Original
    x_data = np.asarray(pd.to_numeric(distance[idx_start:] * 1e3, errors='coerce'))     # [km] -> [m]
    s_data = np.asarray(pd.to_numeric(surface[idx_start:], errors='coerce'))            # [m]
    b_data = np.asarray(pd.to_numeric(-bed[idx_start:], errors='coerce'))               # [m]
    f_data = np.asarray(pd.to_numeric(-float_base[idx_start:], errors='coerce'))        # [m]
    f_interp = np.asarray(pd.to_numeric(-interp_base[idx_start:], errors='coerce'))     # [m]
    vel_data = np.asarray(pd.to_numeric(velocity[idx_start:], errors='coerce'))        # [m/s]


    '''
    x_data = pd.to_numeric(distance[idx_start:] * 1e3, errors='coerce')[:50]     # [km] -> [m]
    s_data = pd.to_numeric(surface[idx_start:], errors='coerce')[:50]            # [m]
    b_data = pd.to_numeric(-bed[idx_start:], errors='coerce')[:50]               # [m]
    f_data = pd.to_numeric(-float_base[idx_start:], errors='coerce')[:50]        # [m]
    f_interp = pd.to_numeric(-interp_base[idx_start:], errors='coerce')[:50]     # [m]
    '''


    Nx = len(x_data)

    # # Grounding point
    # grounding_data_idx_guess = 68      # 81?                 # ??? Why 68 ???
    # idx_gr = grounding_data_idx_guess + 1 - idx_start

    # if idx_gr <= len(x_data):
    #     x_gr = x_data[idx_gr]

    # v_gr = x_data >= x_gr                       # Grounded region
    # v_sb = x_data < x_gr                     # Submerged region

    # Ice thickness profile from the bed topopgraphy, measured surface topography, and estimated ice floor topgraphy (assuming floatation)
    h_data = s_data + np.fmin(f_data, b_data)

    # Centreline profile based on the data
    # zm_data = s_data - h_data / 2               # Only used in the grounded region, where h_data is known
    zm_data = (h_data / 2) - b_data

    # Parameters
    # E_ice = 0.32e3
    # E_ice = 2e5
    # E_ice = 0.32e9                              # [Pa]
    # E_ice = 0.5e9                              # [Pa]
    # E_ice = 3.9e9                              # [Pa]
    E_ice = 0.88e9                               # Vaughan, D. G. (1995). Tidal flexure at ice shelf margins. Journal of Geophysical Research: Solid Earth, 100(B4), 6213-6224.

    nu_ice = 0.3                                # [Pa.s]
    # rho_w = 1e3                                 # [kg/m^3]
    # rho_i = 0.8788 * rho_w                      # [kg/m^3]

    rho_w = 1028                                 # [kg/m^3]
    rho_i = 878.8                      # [kg/m^3]
    g = 9.81                                    # [m/s^2]

    # k0 = 0.5e2 * rho_i * g                        # [Pa/m]
    k0 = 1e2 * rho_i * g                        # [Pa/m]
    # k0 = 1e3 * rho_i * g                        # [Pa/m]
    # k0 = 1e4 * rho_i * g                        # [Pa/m]
    # k0 = 1e5 * rho_i * g                        # [Pa/m]
    # k0 = 1e8 * rho_i * g                        # [Pa/m]

    # Bending stiffness without h^3 part
    gamma_val = E_ice / (12 * (1 - nu_ice**2))  # [Pa]

    # Coefficients for use in the ODE for the centreline profile
    C1 = gamma_val / (rho_w * g)                # [m]
    C2 = k0 / (rho_w * g)                       # [  ]
    C3 = rho_i / rho_w                          # [  ]


    def ice_profile(x, s, b, f, f_interp, zm, h, b0, title):
        lgd_flag = 1
        
        fig, ax = plt.subplots(3, 1, figsize=(12, 6), gridspec_kw={'height_ratios': [3, 1, 1]})
        ax[0].set_title(title)
        ax[0].axhline(0, color='k', linestyle=':')
        ax[0].plot(x, -b, label="Compressed bed", color='#A9561E')
        ax[0].fill_between(x, -b, -1000, color='#A9561E', alpha=0.3, interpolate=True)
        ax[0].plot(x, -b0, label="Uncompressed bed", color='#A9561E', linestyle="--")
        ax[0].plot(x, s, label="Surface", color="b")
        ax[0].plot(x, -f, label="Ice base (float)", color="purple")
        # ax[0].axvline(x_data[64] - 3000, color='k', linestyle=':')
        ax[0].plot(x, -f_interp, label="Ice base (interp.)", color="green")
        ax[0].plot(x, zm, "k--", label="$z_m$")
        ax[0].plot(x, zm - h/2, "k-.", label="$z_m - h/2$")
        ax[0].plot(x, zm + h/2, "k-.", label="$z_m + h/2$")
        ax[0].fill_between(x, zm + h/2, zm - h/2, color='#7DF9FF', alpha=0.3, interpolate=True)
        ax[0].fill_between(x, zm - h/2, -b0, where=((zm - h/2) > -b0), color='C0', alpha=0.4, interpolate=True)
        if lgd_flag:
            ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax[0].set_ylabel("$z$ (m)")
        # ax[0].set_xlim(400, 20200)

        ax[1].axhline(0, color="k", linestyle=":")
        ax[1].plot(x, h, "r.-", label='h')
        ax[1].set_xlabel("$x$ (m)")
        ax[1].set_ylabel("$h$ (m)")
        ax[1].plot(x, s_data + b_data, "b.-", label='s+b')
        # ax[1].set_xlim(400, 20200)
        ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

        # Ice mass loss
        dh_dx = polyfit_data_gradient(x, h, 1, 4, 3)
        total_ice_mass_change = np.sum(cumulative_trapezoid((-rho_i * dh_dx[~np.isnan(f_data)] * vel_data[~np.isnan(f_data)]), x[~np.isnan(f_data)]))
        profile_length = x_data[len(x_data[~np.isnan(f_data)])]
        print(profile_length)

        # ax[2].axhline(0, color="k", linestyle=":")
        ax[2].plot(x, (-rho_i * dh_dx * vel_data), "k.-", label='Ice mass loss')
        ax[2].set_xlabel("$x$ (m)")
        ax[2].set_ylabel("Accumulation/\nablation (kg/yr)")
        ax[2].set_title(f"Total accumulation/ablation of ice mass = {(total_ice_mass_change/(1e6 * profile_length)):.3E}" + r"Gt$yr^{-1}$$km^{-2}$")
        # ax[2].set_xlim(400, 20200)
        ax[2].legend(loc='center left', bbox_to_anchor=(1, 0.5))

        fig.suptitle('Ice sheet profile', fontsize=16)

        plt.tight_layout()
        plt.show()

    def update_profile(p_init, df_dp, f_current, f_current_data, p_data, v_change, omega):

        # Calculate the Newton-Raphson update
        p_update = p_init - ((f_current - f_current_data) / df_dp)

        # Apply updates only where allowed; otherwise, use known data
        p_update[~v_change] = p_data[~v_change]

        # Apply a fraction (omega) of the Newton-Raphson step
        p_new = np.copy(p_data)
        p_new[v_change] = p_init[v_change] + omega * (p_update[v_change] - p_init[v_change])

        return p_new, p_update
    
    def h_update_profile(thickness_init, df_dp, surface_current, surface_data, thickness_data_gr, v_ungrounded, omega):

        # np.where(df_dp > 0, df_dp, -df_dp)
        df_dp_scaled = df_dp / np.max(np.abs(df_dp))  # Normalize to a range [-1, 1]
        # epsilon = 1e-5
        # np.where(df_dp > epsilon, df_dp, -df_dp)


        # Calculate the Newton-Raphson update
        p_update = thickness_init - ((surface_current - surface_data) / df_dp_scaled)

        # Apply updates only where allowed; otherwise, use known data
        p_update[~v_ungrounded] = thickness_data_gr[~v_ungrounded]

        # Apply a fraction (omega) of the Newton-Raphson step
        p_new = np.copy(thickness_data_gr)
        p_new = thickness_init + omega * (p_update - thickness_init)

        return p_new, p_update
    
    def h_update_plot(x, s, b, b0, zm, h_current, h_update, h_new, dzm_dh, omega, lgd_flag, str_title):
        # def contiguous_regions(condition):
        #     condition = np.asarray(condition)
        #     d = np.diff(condition.astype(int))
        #     idxs_start = np.where(d == 1)[0] + 1
        #     idxs_end = np.where(d == -1)[0] + 1
        #     if condition[0]:
        #         idxs_start = np.insert(idxs_start, 0, 0)
        #     if condition[-1]:
        #         idxs_end = np.append(idxs_end, len(condition))
        #     return np.array([idxs_start, idxs_end]).T
        from matplotlib.patches import Rectangle

        def contiguous_regions(v):

            v = np.asarray(v).astype(int)  # Convert boolean to integer (0 or 1)
            N = len(v)
            
            d = np.diff(np.sign(v))  # Difference between consecutive elements

            # Identify the start indices of regions where v > 0
            idxs_start = np.where(d > 0)[0] + 1
            if v[0] > 0:
                idxs_start = np.insert(idxs_start, 0, 0)

            # Identify the end indices of regions where v > 0
            idxs_end = np.where(d < 0)[0]
            if v[-1] > 0:
                idxs_end = np.append(idxs_end, N - 1)

            # Combine start and end indices
            idxs = np.column_stack((idxs_start, idxs_end))
            
            return idxs

        def xregion(ax, x_start, x_end, **kwargs):
            ax.add_patch(Rectangle((x_start, -1e5), x_end - x_start, 2e5, **kwargs))
        
        min_dzm_dh = x[np.argmin(dzm_dh)]

        line_width = 1.5

        v_gr = (zm - h_current / 2) <= -b0
        idxs_gr = contiguous_regions(v_gr)

        # fig, axs = plt.subplots(7, 1, figsize=(12, 10), constrained_layout=True)
        fig, axs = plt.subplot_mosaic([[0],[0],[0],[3],[4],[5],[6],[7]], figsize=(12, 10), constrained_layout=True)
        fig.suptitle(str_title, fontsize=18)

        # Plot 1: Glacier profiles and topography
        ax = axs[0]
        ax.set_title("Glacier profiles and topography")
        # for start, end in idxs_gr:
        #     xregion(ax, x[start], x[end], color='blue', alpha=0.05, label='Grounded region')
        ax.axhline(0, color='k', linestyle=':', label='Sea level')
        ax.plot(x, -b, '-', color=[0.635, 0.078, 0.184], linewidth=line_width, label='Compressed bed')
        ax.plot(x, -b0, '--', color=[0.635, 0.078, 0.184], linewidth=line_width, label='Uncompressed bed')
        ax.plot(x, s, '-', color=[0, 0.447, 0.741], linewidth=line_width, label='Surface')
        ax.plot(x, zm, 'k--', linewidth=line_width, label='$z_m$')
        ax.plot(x, zm - h_current / 2, 'k-.', linewidth=line_width, label='$z_m - h/2$')
        ax.plot(x, zm + h_current / 2, 'k-.', linewidth=line_width, label='$z_m + h/2$')
        ax.plot(x, zm - h_new / 2, 'k:', linewidth=line_width, label=r'zm +/- $h_{new}$/2')
        ax.plot(x, zm + h_new / 2, 'k:', linewidth=line_width)
        ax.set_ylabel('$z$ (m)', fontsize=16)

        ax.fill_between(x, -b0, -1050, color='#A9561E', alpha=0.3, interpolate=True)
        ax.fill_between(x, zm + h_current / 2, zm - h_current / 2, color='#7DF9FF', alpha=0.3, interpolate=True)
        ax.fill_between(x, zm - h_current / 2, -b0, where=((zm - h_current / 2) > -b0), color='C0', alpha=0.4, interpolate=True)

        # if lgd_flag:
        ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))
        
        # Plot 2: Current ice thickness
        ax = axs[3]
        ax.set_title("Ice thickness")
        # for start, end in idxs_gr:
        #     xregion(ax, x[start], x[end], color='blue', alpha=0.05)
        ax.axvline(min_dzm_dh, color='magenta', linestyle='-')
        ax.plot(x, h_current, label='$h_{current}$')
        ax.set_ylabel('$h_{current}$ (m)', fontsize=12)
        ax.grid(True)
        ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))

        ax = axs[4]
        ax.set_title("Ice thickness sensitivity profile (dzm/dh)")
        # for start, end in idxs_gr:
        #     xregion(ax, x[start], x[end], color='blue', alpha=0.05)
        ax.axvline(min_dzm_dh, color='magenta', linestyle='-', label=r'Minimum $\frac{dz_m}{dh}$')
        ax.set_ylabel(r'$\frac{dz_m}{dh}$ + 1/2 (m)', fontsize=12)
        ax.plot(x, dzm_dh + 1/2, label='$h_{sensitivity}$')
        ax.grid(True)
        ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))

        ax = axs[5]
        ax.set_title("Ice thickness update")
        # for start, end in idxs_gr:
        #     xregion(ax, x[start], x[end], color='blue', alpha=0.05)
        ax.axvline(min_dzm_dh, color='magenta', linestyle='-')
        ax.plot(x, h_update, linestyle='--', label='$h_{update}$')
        ax.set_ylabel('$Thickness$ (m)', fontsize=12)
        ax.plot(x, h_new, linestyle='-', label='$h_{new}$')
        ax.plot(x, s + b, color='r', linestyle=':', label='$s-b$')
        ax.grid(True)
        ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))

        ax = axs[6]
        ax.set_title("Newton-Raphson update to thickness")
        # for start, end in idxs_gr:
        #     xregion(ax, x[start], x[end], color='blue', alpha=0.05)
        ax.axvline(min_dzm_dh, color='magenta', linestyle='-')
        ax.plot(x, h_update - h_current, linestyle='--', label=r'$\delta h$')
        ax.set_ylabel('h step size (m)', fontsize=12)
        ax.plot(x, (h_update - h_current) * omega, linestyle='-', label=r'$\delta h * omega$')
        ax.grid(True)
        ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))

        ax = axs[7]
        ax.set_title("Submerged/Grounded regions")
        ax.plot(x, (((zm - h_current/2) <= 0).astype(int)), linestyle='--', label=r'$v_{sb}$')
        ax.plot(x, (((zm - h_current/2) <= -b0).astype(int)), linestyle=':', label=r'$v_{gr}$')
        ax.set_ylabel('True/False', fontsize=12)
        ax.grid(True)
        ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))

        plt.show()
    

    # ####
    h_init = s_data.copy() + b_data.copy()
    h_init[:len(f_data[~np.isnan(f_data)])] = s_data[:len(f_data[~np.isnan(f_data)])].copy() + f_data[:len(f_data[~np.isnan(f_data)])].copy()
    # h_init = h_data[0] * np.ones(Nx)
    # # zm_init = s_data.copy() - (h_init.copy()/2) # + 100

    heighest_point_idx = np.argmin(b_data)
    zm_init = -b_data.copy() + (h_init.copy()/2)
    zm_init[:heighest_point_idx - 1] = zm_init[heighest_point_idx]
    # zm_init[:62] = zm_init[62]
    ####


    # h_init = h_data[-1] * np.ones(Nx)
    # zm_init = -b_data.copy() + (h_init.copy()/2) # + 100
    # zm_init[:15] = zm_init[:15] + 50

    # h_init = e3_solution_h.copy()
    # zm_init = e3_solution_zm.copy()

    # Uncompressed bed topography
    b0_init = b_data.copy() #- 1e-3             # Using this line as the starting guess for b0, the code seems to run okay
    # b0_init = - (s_data - h_data)     # Using this line the profile breaks

    # Pre-allocate arrays
    h = h_init.copy()
    b0 = b0_init.copy()
    # h_update = h.copy()
    # b0_update = b0.copy()

    # Iterative process to determine h(x)

    # Parameters for the iteration process
    error_threshold = 1e-6
    error_val = 10 * error_threshold

    # Relaxation parameter
    # omega = 0.1
    # omega = 0.25
    # omega = 0.35
    # omega = 1

    max_hb_iter = 1e5
    h_min = 200
    h_max = 1500

    omega_zm = 1
    omega_hb = 0.1
    # omega_h = 0.1

    err_s_data = np.zeros(shape=(int(max_hb_iter),1))

    ################################

    lgd_flag = 1
    str_title = f'Initial profile'
    ice_profile(x_data, s_data, b_data, f_data, f_interp, zm_init, h, b0, str_title)

    ###############


    run_count = 1

    while (error_val > error_threshold) & (run_count < max_hb_iter):

        # Calculate centreline and sensitivity profiles
        zm, dzm_dh, dzm_db0 = Flexure_profile_unfixed_grounding_line(zm_init, h_init, b0_init, x_data, C1, C2, C3, omega_zm, quiet, force_balance_flag)
        
        v_gr = (zm - h/2) <= -b0                                # Grounded region
        v_sb = ((zm - h/2) <= 0) # & ((zm - h/2) > -b0)           # Submerged region

        # Calculate the updated ice thickness (Newton Raphson)

        # h_update = h_init - (zm + h_init/2 - s_data) / (dzm_dh + 1/2)           # Why plus 1/2 (Is it okay that I adjusted this to 1 to avoid an error)
        # print(h_update, dzm_dh, ((zm + h_init/2 - s_data) / (dzm_dh + 1/2)))    # Used for debugging
        # h_update = h_init - (zm + h_init/2 - s_data) / (dzm_dh + 1)           # Why not plus 1? --> Change the regularisation

        # regularization_factor = 0.1
        # smoothed_dzm_dh = (dzm_dh + 1/2) + regularization_factor * np.gradient(h_init)

        # adaptive_omega = omega_hb / (1 + np.abs(dzm_dh))

        # dzm_dh_v2 = polyfit_data_gradient(h, zm, 1, 4, 3)
        # dzm_db_v2 = polyfit_data_gradient(b0, zm, 1, 4, 3)

        h, h_update = h_update_profile(h_init , dzm_dh + 1/2, zm + (h_init/2), s_data, h_data, ~v_gr, omega_hb)
        b0, b0_update = update_profile(b0_init, dzm_db0, zm, zm_data, b_data, v_gr, omega_hb)

        # Avoid negative values of h
        h[h < h_min] = h_min
        h[h > h_max] = h_max

        # b0[-b0 > (zm - h/2)] = -(zm - h/2)[-b0 > (zm - h/2)]
        
        # map(lambda x: x if x <= 1e-6 else 1e-6, dzm_db0)     # Added this to prevent the infinite error from np.linalg.norm(b0_update - b0_init)
        # dzm_db0 = np.where(dzm_db0 == 0, np.finfo(float).eps, dzm_db0)        # This was another way of achieving the same as the above

        # I've added damping and regularization terms to the newton-raphson below, but these both help but don't solve the problem of the error plateauing to a finite value
        # damping = 1e-3
        # regularization = 0.03
        # b0_update = b0_init - damping * ((zm - zm_data) / (2*dzm_db0 + regularization))

        # b0_update = b0_init - ((zm - zm_data) / (dzm_db0))
        
        # print(f'first and last values of dzm_db0 = {dzm_db0[0]}, {dzm_db0[-1]}')      # Used for degugging

        # Tempered relaxation in relevant regions
        # h[v_sb] = h_init[v_sb] + omega * (h_update[v_sb] - h_init[v_sb])
        # h[v_gr] = h_data[v_gr]                                                  # Ice thickness is known in the grounded regions

        # b0 = b_data                                                             # Bed topography is known in the ungrounded regions
        # b0[v_gr] = b0_init[v_gr] + omega * (b0_update[v_gr] - b0_init[v_gr])

        # previous_error = error_val
        error_val = (1/(2*Nx)) * (np.linalg.norm(h_init - h) + np.linalg.norm((b0_init - b0)))     # This keeps plateauing to a finite error that is > error_threshold

        err_s_data[run_count+1] = (1/(Nx)) * np.linalg.norm(zm + (h/2) - s_data)

        # print(b0 - b0_init)
        # print(h - h_init)
        
        str_title = 'Run {}: error = {:.3e}'.format(run_count, error_val)
        lgd_flag = 0
        # Plot showing how h is being updated at this stage
        if run_count % fig_freq == 0:
            h_update_plot(x_data, s_data, b_data, b0, zm, h_init, h_update, h, dzm_dh, omega_hb, lgd_flag, str_title)

        # Plot showing how b0 is being updated at this stage
        # if run_count % 10 == 0:
            # b0_update_plot(x_data, s_data, b_data, h, zm, b0_init, b0_update, b0, dzm_db0, omega_hb, lgd_flag, str_title)
        

        # Comment these two lines out if you don't want to see the profile as it's calculated, speeding up the iterative process

        if (run_count % fig_freq == 0):                                               # Plotting every 50th iteration
                run_title = f"Run {run_count}: error = {error_val:.3e}"
                ice_profile(x_data, s_data, b_data, f_data, f_interp, zm, h, b0, run_title)

        # run_title = f"Run {run_count}: error = {error_val:.3e}"
        # ice_profile(x_data, s_data, b_data, f_data, f_interp, zm, h, b0, run_title)
        
        # Use the results as the starting point for the next iteration
        h_init = h.copy()
        b0_init = b0.copy()

        run_count += 1

        # print('np.linalg.norm(h_update - h_init) =', np.linalg.norm(h_update - h_init))     # Used for debugging
        
        # The vector norm of the bottom iteration is currently is much larger than the above line for the top surface
        if (run_count % 10 == 0):           
            print('np.linalg.norm(b0 - b0_init)', np.linalg.norm(b0 - b0_init))   # Used for debugging
            print(f"Run {run_count}: error = {error_val:.3e}")

        # This is a very crude temporary error check to preventing the code iterating forever
        # It doesn't iterate until the errror is small anymore, but I've made it break the loop when the error stops changing
        # if np.abs(previous_error - error_val) / previous_error < error_threshold:
            # break

    # Plot the final results against the Thwaites data
    final_title = f"$E/(12 (1- \N{GREEK SMALL LETTER NU}^2) \N{GREEK SMALL LETTER RHO}_w g) =${C1:.3g}m, $k_0 / \N{GREEK SMALL LETTER RHO}_w g =$ {C2:.3g}, $\N{GREEK SMALL LETTER RHO}_i / \N{GREEK SMALL LETTER RHO}_w$ = {C3:.3g}"
    ice_profile(x_data, s_data, b_data, f_data, f_interp, zm, h_init, b0_init, final_title)
