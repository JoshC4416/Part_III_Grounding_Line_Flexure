import numpy as np
from math import factorial
from scipy.special import comb
from numpy.polynomial.polynomial import Polynomial
from numpy.polynomial import Polynomial as P


def polyfit_data_gradient(x, F, n_der, n_poly, q):
    
    # Ensure valid stencil and polynomial degree
    if 2 * q < n_poly:
        raise ValueError(f"Stencil parameter must be >= {np.ceil(n_poly / 2)} for this order of polynomial!")
    
    if n_der > n_poly:
        raise ValueError("The order of the polynomial is too low to calculate this derivative!")
    
    # Initialize derivative array
    dF = np.zeros_like(x)
    N = len(x)

    # Helper function to compute derivative    
    def derivative_calc(x, F, idx, m, n_der, n_poly):
        
        v = x[idx] - x[m]  # Center the x-values around the current point
        Y = F[idx]

        # Manually scale and shift v for stability
        mu_shift = np.mean(v)
        mu_scale = np.std(v)
        v_scaled = (v - mu_shift) / mu_scale

        # Fit polynomial to scaled data
        p_hat = np.polyfit(v_scaled, Y, n_poly)

        # Calculate the derivative value
        der_val = 0
        for i in range(n_der, n_poly + 1):
            der_val += comb(i, i - n_der) * ((-mu_shift / mu_scale) ** (i - n_der)) * (mu_scale ** -i)

        der_val *= factorial(n_der) * p_hat[n_poly - n_der]
        return der_val

    # The bulk of the domain
    for m in range(q, N - q):
        idx = np.arange(m - q, m + q + 1)
        dF[m] = derivative_calc(x, F, idx, m, n_der, n_poly)

    # Fill in the first q rows (Look Right)
    for m in range(q):
        idx = np.arange(2 * q + 1)
        dF[m] = derivative_calc(x, F, idx, m, n_der, n_poly)

    # Fill in the final q rows (Look Left)
    for m in range(N - q, N):
        idx = np.arange(N - 2 * q, N)
        dF[m] = derivative_calc(x, F, idx, m, n_der, n_poly)

    return dF
