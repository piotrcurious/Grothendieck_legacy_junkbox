"""
Algebraic Geometry and Combinatorics of Projective Quadrics
============================================================
This module implements exact algebraic geometry and combinatorial operations
for Gegenbauer polynomials and complex projective quadric hypersurfaces Q_{d-2} c P^{d-1}:
1. Hilbert Polynomial h^0(Q_{d-2}, O(n)) via ideal short exact sequence on P^{d-1}
2. Normalized Jacobi Recurrence Coefficients for Gelfand algebra multiplication phi_1 * phi_n
3. Pochhammer Symbols and Hypergeometric Series Expansion Coefficients
"""

import math
from typing import List, Tuple


def quadric_hilbert_polynomial(d: int, n: int) -> int:
    """
    Computes dim H^0(Q_{d-2}, O(n)) for complex projective quadric Q_{d-2} c P^{d-1}
    defined by z_1^2 + ... + z_d^2 = 0 via the short exact sequence:
      0 -> O_{P^{d-1}}(n-2) -> O_{P^{d-1}}(n) -> O_{Q_{d-2}}(n) -> 0

    Formula: binom(n+d-1, d-1) - binom(n+d-3, d-1) = [(n+lambda)/lambda] * binom(n+2*lambda-1, n)
    """
    if d < 3 or n < 0:
        raise ValueError("Sphere dimension d must be >= 3 and degree n >= 0.")
    if n == 0:
        return 1
    return math.comb(n + d - 1, d - 1) - math.comb(n + d - 3, d - 1)


def pieri_coefficients(n: int, lambda_val: float) -> Tuple[float, float]:
    """
    Computes unnormalized Pieri rule coefficients for C_n^(lambda):
      x * C_n^(lambda) = C_+ * C_{n+1}^(lambda) + C_- * C_{n-1}^(lambda)
    """
    c_plus = (n + 1.0) / (2.0 * (n + lambda_val))
    c_minus = (n + 2.0 * lambda_val - 1.0) / (2.0 * (n + lambda_val))
    return c_plus, c_minus


def normalized_jacobi_coefficients(n: int, lambda_val: float) -> Tuple[float, float]:
    """
    Computes exact normalized Jacobi recurrence coefficients for zonal functions phi_n:
      phi_1(x) * phi_n(x) = a_n * phi_{n+1}(x) + b_n * phi_{n-1}(x)
    where:
      a_n = (n + 2*lambda) / (2 * (n + lambda))
      b_n = n / (2 * (n + lambda))
      a_n + b_n = 1.0
    """
    a_n = (n + 2.0 * lambda_val) / (2.0 * (n + lambda_val))
    b_n = n / (2.0 * (n + lambda_val))
    return a_n, b_n


def pochhammer(a: float, k: int) -> float:
    """Computes the rising Pochhammer symbol (a)_k = a * (a+1) * ... * (a+k-1)."""
    if k < 0:
        raise ValueError("k must be non-negative.")
    val = 1.0
    for i in range(k):
        val *= (a + i)
    return val


def schubert_intersection_coefficients(n: int, lambda_val: float) -> List[float]:
    """
    Computes the exact hypergeometric series coefficients:
      c_k = [(-1)^k * binom(n, k) * (n + 2*lambda)_k] / (lambda + 0.5)_k
    where C_n^(lambda)(x) = binom(n+2*lambda-1, n) * sum_{k=0}^n c_k * ((1-x)/2)^k.
    Note: No extra k! in the denominator because (-n)_k / k! = (-1)^k * binom(n, k).
    """
    coeffs = []
    for k in range(n + 1):
        num = ((-1.0) ** k) * math.comb(n, k) * pochhammer(n + 2.0 * lambda_val, k)
        den = pochhammer(lambda_val + 0.5, k)
        coeffs.append(num / den)
    return coeffs


if __name__ == "__main__":
    print("--- ALGEBRAIC GEOMETRY & COMBINATORICS DEMO ---")
    d_dim = 5  # d=5 -> Lambda=1.5, Q_3 c P^4
    n_deg = 4
    lambda_p = (d_dim - 2) / 2.0

    h0 = quadric_hilbert_polynomial(d_dim, n_deg)
    a_n, b_n = normalized_jacobi_coefficients(n_deg, lambda_p)
    schubert = schubert_intersection_coefficients(n_deg, lambda_p)

    print(f"Projective Quadric Q_{d_dim-2} c P^{d_dim-1}, degree n={n_deg}:")
    print(f"  * Hilbert Polynomial dim H^0(O_Q({n_deg})): {h0}")
    print(f"  * Normalized Jacobi Recurrence Coefficients (a_n, b_n): ({a_n:.6f}, {b_n:.6f}) [a_n + b_n = {a_n + b_n:.1f}]")
    print(f"  * Exact Hypergeometric Coefficients: {[round(c, 4) for c in schubert]}")
