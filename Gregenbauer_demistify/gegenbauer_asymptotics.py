"""
Numerical Asymptotics and Verification of Gegenbauer Polynomials
================================================================
This module implements exact evaluation and asymptotic approximations
for Gegenbauer polynomials C_n^{(lambda)}(x) based on the representation-theoretic
and semiclassical framework:
1. Exact Evaluation via scipy.special / mpmath
2. Interior WKB / Weyl Semiclassical Expansion
3. Endpoint Mehler-Heine Bessel Boundary-Layer Expansion
4. Composite Matched Asymptotic Expansion
"""

import numpy as np
from scipy.special import eval_gegenbauer, gamma, gammaln, jv


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
    return np.exp(log_c_n_1(n, lambda_val))


def interior_wkb_approx(n: int, lambda_val: float, theta: np.ndarray) -> np.ndarray:
    """
    Interior WKB / Weyl Semiclassical Approximation for C_n^(lambda)(cos(theta)).

    Formula:
      C_n^(lambda)(cos(theta)) ~ [2^(1-lambda) / Gamma(lambda)] * n^(lambda-1) * (sin(theta))^(-lambda)
                                 * cos((n + lambda)*theta - lambda * pi / 2)
    """
    K = n + lambda_val
    coeff = (2.0 ** (1.0 - lambda_val) / gamma(lambda_val)) * (n ** (lambda_val - 1.0))
    amplitude = (np.sin(theta)) ** (-lambda_val)
    phase = K * theta - (lambda_val * np.pi / 2.0)
    return coeff * amplitude * np.cos(phase)


def mehler_heine_bessel_approx(n: int, lambda_val: float, theta: np.ndarray) -> np.ndarray:
    """
    Endpoint Mehler-Heine Bessel Boundary-Layer Approximation for C_n^(lambda)(cos(theta)).

    Formula:
      C_n^(lambda)(cos(theta)) ~ C_n^(lambda)(1) * Cal_J_{lambda-1/2}((n + lambda)*theta)
    where:
      C_n^(lambda)(1) = binom(n + 2*lambda - 1, n)
      Cal_J_{nu}(z) = 2^{nu} * Gamma(nu + 1) * z^{-nu} * J_{nu}(z)
    """
    K = n + lambda_val
    z = K * theta
    nu = lambda_val - 0.5

    # C_n^(lambda)(1) computed stably
    c_n_1 = c_n_1_val(n, lambda_val)

    # Normalized Bessel kernel Cal_J_nu(z)
    # Handle z -> 0 limit gracefully
    z_safe = np.where(z == 0, 1e-15, z)
    cal_j_nu = (2.0 ** nu) * gamma(nu + 1.0) * (z_safe ** (-nu)) * jv(nu, z_safe)
    cal_j_nu = np.where(z == 0, 1.0, cal_j_nu)

    return c_n_1 * cal_j_nu


def composite_matched_approx(n: int, lambda_val: float, theta: np.ndarray) -> np.ndarray:
    """
    Composite Matched Asymptotic Approximation valid uniformly across [0, pi - epsilon].
    Combines Bessel endpoint and WKB interior by adding them and subtracting their matching term.
    """
    K = n + lambda_val
    z = K * theta
    nu = lambda_val - 0.5
    c_n_1 = c_n_1_val(n, lambda_val)

    # Bessel term
    bessel_term = mehler_heine_bessel_approx(n, lambda_val, theta)

    # WKB term
    wkb_term = interior_wkb_approx(n, lambda_val, theta)

    # Overlap / Matching term:
    # Large-z limit of Bessel / Small-theta limit of WKB
    z_safe = np.where(z == 0, 1e-15, z)
    matching_term = (c_n_1 * (2.0 ** nu) * gamma(nu + 1.0) * (z_safe ** (-nu)) *
                     np.sqrt(2.0 / (np.pi * z_safe)) * np.cos(z_safe - lambda_val * np.pi / 2.0))

    return bessel_term + wkb_term - matching_term


def verify_asymptotic_convergence(n_list=[50, 100, 200, 400], lambda_val=1.5):
    """
    Verify numerical convergence rates of interior WKB and endpoint Bessel approximations
    as n increases.
    """
    print(f"\n--- ASYMPTOTIC CONVERGENCE VERIFICATION (lambda = {lambda_val}) ---")

    # 1. Interior Test (theta = pi/4)
    theta_int = np.pi / 4.0
    x_int = np.cos(theta_int)
    print("\n[Interior Regime Test at theta = pi/4]")
    print(f"{'n':>6} | {'Exact C_n':>14} | {'WKB Approx':>14} | {'Rel Error':>12}")
    print("-" * 55)
    for n in n_list:
        exact = exact_gegenbauer(n, lambda_val, x_int)
        wkb = interior_wkb_approx(n, lambda_val, np.array([theta_int]))[0]
        rel_err = abs(exact - wkb) / abs(exact)
        print(f"{n:6d} | {exact:14.6f} | {wkb:14.6f} | {rel_err:12.6e}")

    # 2. Endpoint Test (z = 2.5 fixed, theta = z / (n + lambda))
    z_fix = 2.5
    print("\n[Endpoint Boundary Layer Test at z = 2.5]")
    print(f"{'n':>6} | {'Exact / C_n(1)':>16} | {'Bessel Kernel':>16} | {'Abs Error':>12}")
    print("-" * 60)
    for n in n_list:
        K = n + lambda_val
        theta_end = z_fix / K
        x_end = np.cos(theta_end)
        exact = exact_gegenbauer(n, lambda_val, x_end)
        c_n_1 = c_n_1_val(n, lambda_val)
        exact_ratio = exact / c_n_1

        bessel = mehler_heine_bessel_approx(n, lambda_val, np.array([theta_end]))[0] / c_n_1
        abs_err = abs(exact_ratio - bessel)
        print(f"{n:6d} | {exact_ratio:16.6f} | {bessel:16.6f} | {abs_err:12.6e}")


if __name__ == "__main__":
    verify_asymptotic_convergence()
