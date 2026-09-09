"""
Algebraic Geometry and Combinatorics of Projective Quadrics
============================================================
This module implements exact algebraic geometry and combinatorial operations
for Gegenbauer polynomials and complex projective quadric hypersurfaces Q_{d-2} c P^{d-1}:
1. Hilbert Polynomial h^0(Q_{d-2}, O(n)) via ideal short exact sequence on P^{d-1}
2. Pieri Rule Intersection Products for tensor decomposition V_1 x V_n -> V_{n+1} + V_{n-1}
3. Pochhammer Symbols and Schubert Cycle Intersection Numbers
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
    Computes Pieri rule intersection coefficients for section bundle multiplication
    x * C_n^(lambda) = C_+ * C_{n+1}^(lambda) + C_- * C_{n-1}^(lambda)
    corresponding to tensor product decomposition V_1 x V_n -> V_{n+1} + V_{n-1}.
    """
    c_plus = (n + 1.0) / (2.0 * (n + lambda_val))
    c_minus = (n + 2.0 * lambda_val - 1.0) / (2.0 * (n + lambda_val))
    return c_plus, c_minus


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
    Computes the hypergeometric Schubert cycle intersection coefficients:
      c_k = [(-1)^k * binom(n, k) * (n + 2*lambda)_k] / [k! * (lambda + 0.5)_k]
    which represent intersection counts in the Schubert filtration of Gr(1, Q_{d-2}).
    """
    coeffs = []
    for k in range(n + 1):
        num = ((-1.0) ** k) * math.comb(n, k) * pochhammer(n + 2.0 * lambda_val, k)
        den = math.factorial(k) * pochhammer(lambda_val + 0.5, k)
        coeffs.append(num / den)
    return coeffs


if __name__ == "__main__":
    print("--- ALGEBRAIC GEOMETRY & COMBINATORICS DEMO ---")
    d_dim = 5  # d=5 -> Lambda=1.5, Q_3 c P^4
    n_deg = 4
    lambda_p = (d_dim - 2) / 2.0

    h0 = quadric_hilbert_polynomial(d_dim, n_deg)
    c_p, c_m = pieri_coefficients(n_deg, lambda_p)
    schubert = schubert_intersection_coefficients(n_deg, lambda_p)

    print(f"Projective Quadric Q_{d_dim-2} c P^{d_dim-1}, degree n={n_deg}:")
    print(f"  * Hilbert Polynomial dim H^0(O_Q({n_deg})): {h0}")
    print(f"  * Pieri Rule Coefficients (C_+, C_-): ({c_p:.6f}, {c_m:.6f})")
    print(f"  * Schubert Cycle Intersection Coefficients: {[round(c, 4) for c in schubert]}")
