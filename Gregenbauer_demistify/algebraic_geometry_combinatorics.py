"""
Algebraic Geometry and Combinatorics of Projective Quadrics
============================================================
This module implements exact algebraic geometry and combinatorial operations
for Gegenbauer polynomials and complex projective quadric hypersurfaces Q_{d-2} c P^{d-1}:
1. Hilbert Polynomial h^0(Q_{d-2}, O(n)) via Hilbert series H_{R(Q)}(t) = (1-t^2)/(1-t)^d
2. Quotient Ring Normal Forms: Polynomial remainder modulo q = sum(z_i^2) in C[z_1,...,z_d]/(q)
   Note: Normal form reduction convention maps z_d^2 -> -(z_1^2 + ... + z_{d-1}^2)
3. Normalized Jacobi Recurrence Coefficients for Gelfand algebra multiplication M_x: phi_n -> x * phi_n
4. Pochhammer Symbols and Hypergeometric Series Expansion Coefficients
"""

import math
from typing import Dict, Tuple, List


def quadric_hilbert_series_dim(d: int, n: int) -> int:
    """
    Computes dim R(Q)_n = dim H^0(Q_{d-2}, O(n)) for complex projective quadric Q_{d-2} c P^{d-1}
    defined by z_1^2 + ... + z_d^2 = 0 via the Hilbert series:
      H_{R(Q)}(t) = (1 - t^2) / (1 - t)^d = sum_{n=0}^inf (dim R(Q)_n) t^n

    Formula: binom(n+d-1, d-1) - binom(n+d-3, d-1)
    """
    if d < 3 or n < 0:
        raise ValueError("Sphere dimension d must be >= 3 and degree n >= 0.")
    if n == 0:
        return 1
    return math.comb(n + d - 1, d - 1) - math.comb(n + d - 3, d - 1)


class QuadricQuotientPolynomial:
    """
    Represents a polynomial in C[z_1, ..., z_d] / (q) where q = z_1^2 + ... + z_d^2.
    Monomials are represented as tuples of non-negative integers (a_1, ..., a_d).
    Normal form reduction convention recursively replaces z_d^2 -> -(z_1^2 + ... + z_{d-1}^2).
    """

    def __init__(self, d: int, terms: Dict[Tuple[int, ...], float] = None):
        self.d = d
        self.terms: Dict[Tuple[int, ...], float] = {}
        if terms:
            for exp_tuple, coeff in terms.items():
                if len(exp_tuple) != d:
                    raise ValueError(f"Exponent tuple {exp_tuple} must have length {d}")
                self._add_term(exp_tuple, coeff)

    def _add_term(self, exp_tuple: Tuple[int, ...], coeff: float):
        if abs(coeff) < 1e-12:
            return

        # Check if reduction modulo q = z_1^2 + ... + z_d^2 is needed (using z_d^2 = -(z_1^2 + ... + z_{d-1}^2))
        exp_list = list(exp_tuple)
        if exp_list[-1] >= 2:
            # Replace one instance of z_d^2 with -sum_{i=1}^{d-1} z_i^2 and reduce recursively
            remainder_z_d = exp_list[-1] - 2
            for i in range(self.d - 1):
                new_exp = list(exp_list)
                new_exp[-1] = remainder_z_d
                new_exp[i] += 2
                self._add_term(tuple(new_exp), -coeff)
        else:
            t = tuple(exp_list)
            self.terms[t] = self.terms.get(t, 0.0) + coeff
            if abs(self.terms[t]) < 1e-12:
                del self.terms[t]

    def normal_form(self) -> 'QuadricQuotientPolynomial':
        """Returns the idempotent normal form polynomial in R(Q)."""
        return QuadricQuotientPolynomial(self.d, self.terms)

    def multiply_by_x(self, var_idx: int = 0) -> 'QuadricQuotientPolynomial':
        """Multiplies by variable z_{var_idx+1} (default z_1 = x) in quotient ring R(Q)."""
        res = QuadricQuotientPolynomial(self.d)
        for exp_tuple, coeff in self.terms.items():
            new_exp = list(exp_tuple)
            new_exp[var_idx] += 1
            res._add_term(tuple(new_exp), coeff)
        return res

    def evaluate(self, point: List[float]) -> float:
        """Evaluates normal form polynomial at a given point in R^d."""
        if len(point) != self.d:
            raise ValueError(f"Point length must be {self.d}")
        total = 0.0
        for exp_tuple, coeff in self.terms.items():
            monomial_val = 1.0
            for var_val, exp in zip(point, exp_tuple):
                monomial_val *= (var_val ** exp)
            total += coeff * monomial_val
        return total

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, QuadricQuotientPolynomial):
            return False
        if self.d != other.d:
            return False
        # Compare terms within 1e-10 tolerance
        all_keys = set(self.terms.keys()).union(set(other.terms.keys()))
        for k in all_keys:
            v1 = self.terms.get(k, 0.0)
            v2 = other.terms.get(k, 0.0)
            if abs(v1 - v2) > 1e-10:
                return False
        return True


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
      M_x phi_n = a_n * phi_{n+1} + b_n * phi_{n-1}
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
    Computes exact hypergeometric series expansion coefficients.
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

    h0 = quadric_hilbert_series_dim(d_dim, n_deg)
    a_n, b_n = normalized_jacobi_coefficients(n_deg, lambda_p)

    print(f"Projective Quadric Q_{d_dim-2} c P^{d_dim-1}, degree n={n_deg}:")
    print(f"  * Hilbert Series Dimension dim R(Q)_{n_deg}: {h0}")
    print(f"  * Normalized Jacobi Recurrence Coefficients (a_n, b_n): ({a_n:.6f}, {b_n:.6f}) [a_n + b_n = {a_n + b_n:.1f}]")

    # Test Quotient Ring Normal Form
    poly = QuadricQuotientPolynomial(3, {(0, 0, 2): 1.0})  # z_3^2 in C[z_1, z_2, z_3]/(z_1^2+z_2^2+z_3^2)
    print(f"  * Normal form of z_3^2 modulo q: {poly.terms}")
