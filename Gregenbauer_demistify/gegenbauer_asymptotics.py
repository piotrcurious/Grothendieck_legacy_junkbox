"""
Numerical Asymptotics, Phase Maps, and Verification of Gegenbauer Polynomials
==============================================================================
This module implements exact evaluation, scaled spherical recurrence algorithms,
two-endpoint phase map regime classification, normalized error surface maps, and
asymptotic approximations for Gegenbauer polynomials C_n^{(lambda)}(x) and zonal
spherical functions phi_n(x) on S^{d-1} = SO(d)/SO(d-1) (where lambda = (d-2)/2):

1. Production Scaled Spherical Recurrence for phi_n^{(d)}(theta)
2. Normalized Spherical Derivative Evaluator phi_n^{(k)}(x)
3. Analytic Gegenbauer Derivatives via k-th Order Shift Formula
4. Normalized Interior WKB / Weyl Semiclassical Wave
5. Two-Endpoint Bessel Boundary-Layer Kernels (North z_0 = K*theta, South z_pi = K*(pi-theta))
6. Two-Overlap Composite Matched Asymptotic Expansion
7. Natural Boundary-Layer Phase Map Classifier in (z_0, z_pi) Coordinates
8. Normalized Error Surface Diagram E(n, theta)
9. Theta-Space Orthogonality Quadrature Integral
"""

import numpy as np
from scipy.special import eval_gegenbauer, gamma, gammaln, jv
from scipy.integrate import quad


def exact_gegenbauer(n: int, lambda_val: float, x: np.ndarray) -> np.ndarray:
    """Exact computation of Gegenbauer polynomial C_n^(lambda)(x) via scipy."""
    return eval_gegenbauer(n, lambda_val, x)


def log_c_n_1(n: int, lambda_val: float) -> float:
    """Computes log(C_n^(lambda)(1)) stably using gammaln for lambda > 0."""
    if lambda_val <= 0:
        raise ValueError("lambda_val must be > 0 for spherical Gegenbauer functions")
    return float(gammaln(n + 2.0 * lambda_val) -
                 gammaln(2.0 * lambda_val) -
                 gammaln(n + 1.0))


def c_n_1_val(n: int, lambda_val: float) -> float:
    """Computes C_n^(lambda)(1) value. Note: may overflow for very large n, lambda."""
    return float(np.exp(log_c_n_1(n, lambda_val)))


def gegenbauer_derivative(n: int, lambda_val: float, x: np.ndarray, k: int = 1) -> np.ndarray:
    """
    Computes exact k-th derivative of Gegenbauer polynomial C_n^(lambda)(x) via formula:
      d^k/dx^k C_n^(lambda)(x) = 2^k * (lambda)_k * C_{n-k}^(lambda+k)(x).
    """
    x_arr = np.asarray(x, dtype=np.float64)
    if k < 0:
        raise ValueError("Derivative order k must be >= 0")
    if k == 0:
        return exact_gegenbauer(n, lambda_val, x_arr)
    if k > n:
        return np.zeros_like(x_arr)

    poch = 1.0
    for i in range(k):
        poch *= (lambda_val + i)
    scale = (2.0 ** k) * poch

    return scale * exact_gegenbauer(n - k, lambda_val + k, x_arr)


def normalized_phi_derivative(n: int, lambda_val: float, x: np.ndarray, k: int = 1) -> np.ndarray:
    """
    Computes exact k-th derivative of normalized zonal function phi_n(x) = C_n^(lambda)(x) / C_n^(lambda)(1):
      phi_n^{(k)}(x) = 2^k * (lambda)_k / C_n^(lambda)(1) * C_{n-k}^(lambda+k)(x).
    """
    if k == 0:
        return normalized_phi_recurrence(n, lambda_val, x)
    c1 = c_n_1_val(n, lambda_val)
    return gegenbauer_derivative(n, lambda_val, x, k=k) / c1


def normalized_phi_recurrence(n: int, lambda_val: float, x: np.ndarray) -> np.ndarray:
    """
    Direct normalized recurrence algorithm for zonal spherical function on S^{d-1}:
      phi_n^{(d)}(theta) = C_n^{((d-2)/2)}(cos(theta)) / C_n^{((d-2)/2)}(1)

    Recurrence relation:
      phi_{k+1}(x) = [(2(k + lambda))/(k + 2*lambda)] * x * phi_k(x) - [k / (k + 2*lambda)] * phi_{k-1}(x)
    Initialized by phi_0(x) = 1, phi_1(x) = x.
    Avoids floating-point amplitude overflow for large n.
    """
    if n < 0 or int(n) != n:
        raise ValueError("n must be a non-negative integer")
    if lambda_val <= 0:
        raise ValueError("lambda_val must be > 0 for spherical Gegenbauer functions")

    x_arr = np.asarray(x, dtype=np.float64)
    if n == 0:
        return np.ones_like(x_arr)
    if n == 1:
        return x_arr.copy()

    phi0 = np.ones_like(x_arr)
    phi1 = x_arr.copy()

    for k in range(1, n):
        a_k = (k + 2.0 * lambda_val) / (2.0 * (k + lambda_val))
        b_k = k / (2.0 * (k + lambda_val))
        phi2 = (x_arr * phi1 - b_k * phi0) / a_k
        phi0, phi1 = phi1, phi2

    return phi1


def log_orthogonality_norm(n: int, lambda_val: float) -> float:
    """
    Computes ln(h_n) stably for the orthogonality norm square:
      int_0^pi [C_n^(lambda)(cos(theta))]^2 sin^{2*lambda}(theta) d_theta = h_n
    """
    return float(np.log(np.pi) +
                 (1.0 - 2.0 * lambda_val) * np.log(2.0) +
                 gammaln(n + 2.0 * lambda_val) -
                 gammaln(n + 1.0) -
                 np.log(n + lambda_val) -
                 2.0 * gammaln(lambda_val))


def orthogonality_norm(n: int, lambda_val: float) -> float:
    """Computes h_n stably using log-space exponential."""
    return float(np.exp(log_orthogonality_norm(n, lambda_val)))


def verify_orthogonality_integral_theta(n: int, m: int, lambda_val: float) -> float:
    """
    Numerically computes the L2 orthogonality integral in theta-space without fractional singularities:
      I_{nm} = int_0^pi C_n^(lambda)(cos(theta)) C_m^(lambda)(cos(theta)) (sin(theta))^(2*lambda) d_theta
    """
    def integrand(theta):
        w = (np.sin(theta)) ** (2.0 * lambda_val)
        x = np.cos(theta)
        return eval_gegenbauer(n, lambda_val, x) * eval_gegenbauer(m, lambda_val, x) * w

    val, _err = quad(integrand, 0.0, np.pi, limit=100)
    return float(val)


def normalized_bessel_kernel(nu: float, z: np.ndarray) -> np.ndarray:
    """
    Normalized Euclidean radial Bessel kernel Cal_J_nu(z) = 2^nu * Gamma(nu + 1) * z^{-nu} * J_nu(z).
    Evaluated stably at z = 0 without artificial epsilons: Cal_J_nu(0) = 1.0.
    """
    z_arr = np.asarray(z, dtype=np.float64)
    out = np.zeros_like(z_arr)
    zero_mask = (np.abs(z_arr) < 1e-14)
    non_zero = ~zero_mask

    out[zero_mask] = 1.0
    if np.any(non_zero):
        z_nz = z_arr[non_zero]
        out[non_zero] = (2.0 ** nu) * gamma(nu + 1.0) * (z_nz ** (-nu)) * jv(nu, z_nz)

    return out


def endpoint_bessel_leading(n: int, lambda_val: float, theta: np.ndarray) -> np.ndarray:
    """
    North-Pole Leading Endpoint Bessel Boundary-Layer Approximation for zonal function phi_n(theta):
      phi_n(theta) ~ Cal_J_{lambda-1/2}((n + lambda)*theta)
    Valid in the local scaling regime z_0 = (n + lambda)*theta = O(1).
    """
    K = n + lambda_val
    z0 = K * np.asarray(theta, dtype=np.float64)
    nu = lambda_val - 0.5
    return normalized_bessel_kernel(nu, z0)


def south_pole_bessel_leading(n: int, lambda_val: float, theta: np.ndarray) -> np.ndarray:
    """
    South-Pole Leading Endpoint Bessel Boundary-Layer Approximation for zonal function phi_n(theta):
      phi_n(theta) ~ (-1)^n * Cal_J_{lambda-1/2}((n + lambda)*(pi - theta))
    Valid in the local scaling regime z_pi = (n + lambda)*(pi - theta) = O(1).
    """
    K = n + lambda_val
    theta_arr = np.asarray(theta, dtype=np.float64)
    z_pi = K * (np.pi - theta_arr)
    nu = lambda_val - 0.5
    parity = (-1.0) ** n
    return parity * normalized_bessel_kernel(nu, z_pi)


def interior_wkb_approx(n: int, lambda_val: float, theta: np.ndarray) -> np.ndarray:
    """
    Normalized Interior WKB / Weyl Semiclassical Wave for zonal spherical function phi_n(theta):
      phi_n(theta) ~ [2^lambda * Gamma(lambda + 1/2) / sqrt(pi)] * cos((n + lambda)*theta - lambda*pi/2) / (N*sin(theta))^lambda
    Valid in the interior oscillatory domain theta >> 1/n and (pi - theta) >> 1/n.
    """
    theta_arr = np.asarray(theta, dtype=np.float64)
    K = n + lambda_val
    coeff = (2.0 ** lambda_val) * gamma(lambda_val + 0.5) / np.sqrt(np.pi)
    sin_theta = np.sin(theta_arr)

    sin_safe = np.where(sin_theta <= 1e-15, 1e-15, sin_theta)
    amplitude = (K * sin_safe) ** (-lambda_val)
    phase = K * theta_arr - (lambda_val * np.pi / 2.0)

    out = coeff * amplitude * np.cos(phase)
    out = np.where((theta_arr <= 1e-15) | (theta_arr >= np.pi - 1e-15), np.nan, out)
    return out


def composite_matched_approx(n: int, lambda_val: float, theta: np.ndarray) -> np.ndarray:
    """
    Two-Endpoint Composite Matched Asymptotic Approximation for zonal function phi_n(theta):
    F_comp = F_north(z_0) + F_south(z_pi) + F_interior(N, theta) - F_{+,overlap}(z_0) - F_{-,overlap}(z_pi).
    Combines North pole Bessel layer (z_0 = K*theta), South pole Bessel layer (z_pi = K*(pi-theta)),
    and Interior WKB wave, subtracting both North and South overlap matching terms.
    Evaluates stable limits at endpoints: phi_n(0) = 1.0 and phi_n(pi) = (-1)^n.
    """
    theta_arr = np.asarray(theta, dtype=np.float64)
    out = np.zeros_like(theta_arr)

    for i, th in enumerate(theta_arr):
        if th <= 1e-14:
            out[i] = 1.0
        elif th >= np.pi - 1e-14:
            out[i] = (-1.0) ** n
        else:
            K = n + lambda_val
            z0 = K * th
            z_pi = K * (np.pi - th)
            nu = lambda_val - 0.5

            bessel_north = normalized_bessel_kernel(nu, np.array([z0]))[0]
            bessel_south = ((-1.0) ** n) * normalized_bessel_kernel(nu, np.array([z_pi]))[0]
            wkb_val = interior_wkb_approx(n, lambda_val, np.array([th]))[0]

            # Matching terms for North and South overlaps
            match_north = (2.0 ** nu) * gamma(nu + 1.0) * (z0 ** (-nu)) * np.sqrt(2.0 / (np.pi * z0)) * np.cos(z0 - lambda_val * np.pi / 2.0)
            match_south = ((-1.0) ** n) * (2.0 ** nu) * gamma(nu + 1.0) * (z_pi ** (-nu)) * np.sqrt(2.0 / (np.pi * z_pi)) * np.cos(z_pi - lambda_val * np.pi / 2.0)

            out[i] = bessel_north + bessel_south + wkb_val - match_north - match_south

    return out


def classify_phase_regime(n: int, lambda_val: float, theta: float, overlap_alpha: float = 0.5) -> str:
    """
    Natural Boundary-Layer Phase Map Classifier on the (z_0, z_pi) coordinates:
      z_0  = (n + lambda) * theta
      z_pi = (n + lambda) * (pi - theta)

    Regimes:
      - 'north_endpoint_bessel': z_0 <= 10.0
      - 'south_endpoint_bessel': z_pi <= 10.0
      - 'overlap_approximation': z_0 <= K^\alpha or z_pi <= K^\alpha
      - 'interior_approximation': z_0 > K^\alpha and z_pi > K^\alpha
    """
    K = n + lambda_val
    z0 = K * theta
    z_pi = K * (np.pi - theta)
    overlap_upper = K ** overlap_alpha

    if z0 <= 10.0:
        return "north_endpoint_bessel"
    elif z_pi <= 10.0:
        return "south_endpoint_bessel"
    elif z0 <= overlap_upper or z_pi <= overlap_upper:
        return "overlap_approximation"
    else:
        return "interior_approximation"


def compute_error_map(n: int, lambda_val: float, num_theta: int = 200) -> dict:
    """
    Generates normalized error surface map E(n, theta) comparing asymptotic approximations
    against the exact scaled spherical recurrence phi_n(theta).
    """
    theta_grid = np.linspace(0.001, np.pi - 0.001, num_theta)
    exact_phi = normalized_phi_recurrence(n, lambda_val, np.cos(theta_grid))

    wkb_phi = interior_wkb_approx(n, lambda_val, theta_grid)
    bessel_north_phi = endpoint_bessel_leading(n, lambda_val, theta_grid)
    bessel_south_phi = south_pole_bessel_leading(n, lambda_val, theta_grid)
    comp_phi = composite_matched_approx(n, lambda_val, theta_grid)

    abs_err_wkb = np.abs(wkb_phi - exact_phi)
    abs_err_comp = np.abs(comp_phi - exact_phi)

    return {
        "theta": theta_grid,
        "exact_phi": exact_phi,
        "wkb_phi": wkb_phi,
        "composite_phi": comp_phi,
        "abs_err_wkb": abs_err_wkb,
        "abs_err_composite": abs_err_comp,
        "max_err_wkb": float(np.nanmax(abs_err_wkb)),
        "max_err_composite": float(np.nanmax(abs_err_comp)),
    }


if __name__ == "__main__":
    print("--- REFACTORED ASYMPTOTICS & TWO-ENDPOINT PHASE MAP DEMO ---")
    n_deg = 200
    lambda_p = 1.5
    theta_val = 0.02

    phi_rec = normalized_phi_recurrence(n_deg, lambda_p, np.array([np.cos(theta_val)]))[0]
    regime = classify_phase_regime(n_deg, lambda_p, theta_val)
    err_map = compute_error_map(n=100, lambda_val=1.5, num_theta=100)

    print(f"Degree n={n_deg}, Lambda={lambda_p}, theta={theta_val}:")
    print(f"  * Normalized Recurrence phi_{n_deg}(cos({theta_val})): {phi_rec:.6f}")
    print(f"  * Natural Phase Map Selection: {regime}")
    print(f"  * Error Map (n=100): Max WKB Error = {err_map['max_err_wkb']:.6e}, Max Composite Error = {err_map['max_err_composite']:.6e}")
