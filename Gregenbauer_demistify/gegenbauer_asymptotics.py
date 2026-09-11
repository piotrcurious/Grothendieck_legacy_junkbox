"""
Numerical Asymptotics, Phase Maps, and Verification of Gegenbauer Polynomials
==============================================================================
This module implements exact evaluation, scaled recurrence algorithms, phase map
regime classification, and asymptotic approximations for Gegenbauer polynomials C_n^{(lambda)}(x)
and zonal spherical functions phi_n(x) on SO(d)/SO(d-1):
1. Production Scaled Recurrence for phi_n(x)
2. Interior WKB / Weyl Semiclassical Expansion
3. Endpoint Mehler-Heine Bessel Boundary-Layer Expansion (Euclidean Kernel)
4. Composite Matched Asymptotic Expansion
5. Operational Phase Map Classifier & Error Diagram E(n, theta)
6. Special Exact Test Anchors (S^2, S^3, S^4)
7. Log-space Orthogonality Norm & Quadrature Integral Verification
"""

import numpy as np
from scipy.special import eval_gegenbauer, gamma, gammaln, jv
from scipy.integrate import quad


def exact_gegenbauer(n: int, lambda_val: float, x: np.ndarray) -> np.ndarray:
    """Exact computation of Gegenbauer polynomial C_n^(lambda)(x)."""
    return eval_gegenbauer(n, lambda_val, x)


def log_c_n_1(n: int, lambda_val: float) -> float:
    """Computes log(C_n^(lambda)(1)) stably using gammaln."""
    return (gammaln(n + 2.0 * lambda_val) -
            gammaln(2.0 * lambda_val) -
            gammaln(n + 1.0))


def c_n_1_val(n: int, lambda_val: float) -> float:
    """Computes C_n^(lambda)(1) stably."""
    return float(np.exp(log_c_n_1(n, lambda_val)))


def normalized_phi_recurrence(n: int, lambda_val: float, x: np.ndarray) -> np.ndarray:
    """
    Direct normalized recurrence algorithm for zonal function phi_n(x) = C_n^(lambda)(x) / C_n^(lambda)(1):
      phi_{k+1}(x) = [(2(k + lambda))/(k + 2*lambda)] * x * phi_k(x) - [k / (k + 2*lambda)] * phi_{k-1}(x)
    Initialized by phi_0(x) = 1, phi_1(x) = x.
    This avoids huge intermediate amplitudes at large n.
    """
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
      int_{-1}^1 C_n^(lambda)(x) C_m^(lambda)(x) (1 - x^2)^(lambda - 1/2) dx = h_n * delta_{nm}
    """
    return (np.log(np.pi) +
            (1.0 - 2.0 * lambda_val) * np.log(2.0) +
            gammaln(n + 2.0 * lambda_val) -
            gammaln(n + 1.0) -
            np.log(n + lambda_val) -
            2.0 * gammaln(lambda_val))


def orthogonality_norm(n: int, lambda_val: float) -> float:
    """Computes h_n stably using log-space exponential."""
    return float(np.exp(log_orthogonality_norm(n, lambda_val)))


def verify_orthogonality_integral(n: int, m: int, lambda_val: float) -> float:
    """
    Numerically computes the L2 orthogonality integral:
      I_{nm} = int_{-1}^1 C_n^(lambda)(x) C_m^(lambda)(x) (1 - x^2)^(lambda - 1/2) dx
    """
    def integrand(x):
        w = (1.0 - x**2) ** (lambda_val - 0.5)
        return eval_gegenbauer(n, lambda_val, x) * eval_gegenbauer(m, lambda_val, x) * w

    val, _err = quad(integrand, -1.0, 1.0, limit=100)
    return val


def normalized_bessel_kernel(nu: float, z: np.ndarray) -> np.ndarray:
    """
    Normalized Euclidean radial Bessel kernel Cal_J_nu(z) = 2^nu * Gamma(nu + 1) * z^{-nu} * J_nu(z).
    Satisfies Cal_J_nu(0) = 1.0.
    """
    z_arr = np.asarray(z, dtype=np.float64)
    z_safe = np.where(z_arr == 0, 1e-15, z_arr)
    cal_j_nu = (2.0 ** nu) * gamma(nu + 1.0) * (z_safe ** (-nu)) * jv(nu, z_safe)
    return np.where(z_arr == 0, 1.0, cal_j_nu)


def mehler_heine_bessel_approx(n: int, lambda_val: float, theta: np.ndarray) -> np.ndarray:
    """
    Endpoint Mehler-Heine Bessel Boundary-Layer Approximation for C_n^(lambda)(cos(theta)).
      C_n^(lambda)(cos(theta)) ~ C_n^(lambda)(1) * Cal_J_{lambda-1/2}((n + lambda)*theta)
    """
    K = n + lambda_val
    z = K * np.asarray(theta, dtype=np.float64)
    nu = lambda_val - 0.5
    c_n_1 = c_n_1_val(n, lambda_val)
    return c_n_1 * normalized_bessel_kernel(nu, z)


def interior_wkb_approx(n: int, lambda_val: float, theta: np.ndarray) -> np.ndarray:
    """
    Interior WKB / Weyl Semiclassical Approximation for C_n^(lambda)(cos(theta)).
    """
    theta_arr = np.asarray(theta, dtype=np.float64)
    K = n + lambda_val
    coeff = (2.0 ** (1.0 - lambda_val) / gamma(lambda_val)) * (n ** (lambda_val - 1.0))
    amplitude = (np.sin(theta_arr)) ** (-lambda_val)
    phase = K * theta_arr - (lambda_val * np.pi / 2.0)
    return coeff * amplitude * np.cos(phase)


def composite_matched_approx(n: int, lambda_val: float, theta: np.ndarray) -> np.ndarray:
    """
    Composite Matched Asymptotic Approximation valid uniformly across [0, pi - epsilon].
    """
    theta_arr = np.asarray(theta, dtype=np.float64)
    K = n + lambda_val
    z = K * theta_arr
    nu = lambda_val - 0.5
    c_n_1 = c_n_1_val(n, lambda_val)

    bessel_term = mehler_heine_bessel_approx(n, lambda_val, theta_arr)
    wkb_term = interior_wkb_approx(n, lambda_val, theta_arr)

    z_safe = np.where(z == 0, 1e-15, z)
    matching_term = (c_n_1 * (2.0 ** nu) * gamma(nu + 1.0) * (z_safe ** (-nu)) *
                     np.sqrt(2.0 / (np.pi * z_safe)) * np.cos(z_safe - lambda_val * np.pi / 2.0))

    return bessel_term + wkb_term - matching_term


def classify_phase_regime(n: int, lambda_val: float, theta: float) -> str:
    """
    Operational Phase Diagram Map for (n, theta)-plane:
      z = (n + lambda) * theta
    Returns selected regime string.
    """
    K = n + lambda_val
    z = K * theta
    sqrtK = np.sqrt(K)

    if n <= 100 or theta > np.pi - 1e-3:
        return "direct_recurrence"
    elif z <= 10.0:
        return "endpoint_approximation"
    elif z <= sqrtK:
        return "overlap_approximation"
    else:
        return "interior_approximation"


def exact_anchor_eval(d: int, n: int, theta: float) -> float:
    """
    Evaluates exact test anchors:
      d=3 (lambda=1/2): P_n(cos(theta))
      d=4 (lambda=1): sin((n+1)theta) / ((n+1)sin(theta))
      d=5 (lambda=3/2): normalized C_n^{(3/2)}(cos(theta))
    """
    x = np.cos(theta)
    if d == 3:
        return float(eval_gegenbauer(n, 0.5, x))
    elif d == 4:
        if abs(theta) < 1e-12:
            return 1.0
        return float(np.sin((n + 1.0) * theta) / ((n + 1.0) * np.sin(theta)))
    elif d == 5:
        return float(normalized_phi_recurrence(n, 1.5, np.array([x]))[0])
    else:
        lambda_val = (d - 2.0) / 2.0
        return float(normalized_phi_recurrence(n, lambda_val, np.array([x]))[0])


if __name__ == "__main__":
    print("--- ASYMPTOTICS & PHASE MAP DEMO ---")
    n_deg = 200
    lambda_p = 1.5
    theta_val = 0.02
    x_val = np.cos(theta_val)

    phi_rec = normalized_phi_recurrence(n_deg, lambda_p, np.array([x_val]))[0]
    regime = classify_phase_regime(n_deg, lambda_p, theta_val)
    anchor_s3 = exact_anchor_eval(d=4, n=10, theta=0.5)

    print(f"Degree n={n_deg}, Lambda={lambda_p}, theta={theta_val}:")
    print(f"  * Normalized Recurrence phi_{n_deg}(cos({theta_val})): {phi_rec:.6f}")
    print(f"  * Phase Map Selection: {regime}")
    print(f"  * Exact Anchor S^3 (d=4, n=10, theta=0.5): {anchor_s3:.6f}")
