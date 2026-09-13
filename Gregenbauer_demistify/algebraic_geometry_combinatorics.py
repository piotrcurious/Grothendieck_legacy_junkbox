"""
Algebraic Geometry and Combinatorics of Projective Quadrics
============================================================
This module implements exact algebraic geometry and combinatorial operations
for Gegenbauer polynomials and complex projective quadric hypersurfaces Q_{d-2} c P^{d-1}:
1. Hilbert Polynomial h^0(Q_{d-2}, O(n)) via Hilbert series H_{R(Q)}(t) = (1-t^2)/(1-t)^d
2. Exact Rational Quotient Ring Normal Forms: Polynomial remainder modulo q = sum(z_i^2) in C[z_1,...,z_d]/(q)
   Note: Normal form reduction convention maps z_d^2 -> -(z_1^2 + ... + z_{d-1}^2)
3. Exact Rational Jacobi Recurrence Coefficients for Gelfand algebra multiplication M_x: phi_n -> x * phi_n
4. Orthonormal Symmetric Jacobi Matrix Coefficients alpha_n = 1/2 * sqrt( (n+1)(n+2*lambda) / ((n+lambda)(n+lambda+1)) )
5. Normalized Gegenbauer _2F_1 Hypergeometric Expansion Coefficients
6. Helper lambda_for_sphere(d) = Fraction(d-2, 2)
"""

from fractions import Fraction
import math
from typing import Dict, List, Tuple, Union

import numpy as np
from scipy.special import gamma

Number = Union[int, Fraction, float]


def lambda_for_sphere(d: int) -> Fraction:
    """Returns exact rational Gegenbauer parameter lambda = (d-2)/2 for sphere S^{d-1}."""
    if d < 3 or int(d) != d:
        raise ValueError("Ambient Euclidean dimension d must be an integer >= 3")
    return Fraction(d - 2, 2)


def quadric_hilbert_series_dim(d: int, n: int) -> int:
    """
    Computes dim R(Q)_n = dim H^0(Q_{d-2}, O(n)) for complex projective quadric Q_{d-2} c P^{d-1}
    defined by z_1^2 + ... + z_d^2 = 0 in ambient dimension d via the Hilbert series:
      H_{R(Q)}(t) = (1 - t^2) / (1 - t)^d = sum_{n=0}^inf (dim R(Q)_n) t^n

    Formula: binom(n+d-1, d-1) - binom(n+d-3, d-1)
    """
    if d < 3 or n < 0:
        raise ValueError("Ambient Euclidean dimension d must be >= 3 and degree n >= 0.")
    if n == 0:
        return 1
    return math.comb(n + d - 1, d - 1) - math.comb(n + d - 3, d - 1)


class QuadricQuotientPolynomial:
    """
    Represents an exact polynomial in C[z_1, ..., z_d] / (q) where q = z_1^2 + ... + z_d^2.
    Monomials are represented as tuples of non-negative integers (a_1, ..., a_d).
    Coefficients use exact rational arithmetic (fractions.Fraction).
    Canonical normal form invariant: z_d exponent is always <= 1.
    """

    def __init__(self, d: int, terms: Dict[Tuple[int, ...], Number] = None):
        if d < 3 or int(d) != d:
            raise ValueError("Ambient Euclidean dimension d must be an integer >= 3")
        self.d = d
        self.terms: Dict[Tuple[int, ...], Fraction] = {}
        if terms:
            for exp_tuple, coeff in terms.items():
                if len(exp_tuple) != d:
                    raise ValueError(f"Exponent tuple {exp_tuple} must have length {d}")
                self._add_term(exp_tuple, Fraction(coeff))

    def _add_term(self, exp_tuple: Tuple[int, ...], coeff: Fraction):
        if coeff == 0:
            return

        exp_list = list(exp_tuple)
        z_d_exp = exp_list[-1]

        if z_d_exp >= 2:
            m = z_d_exp // 2
            r = z_d_exp % 2
            base_exp = exp_list[:-1]

            # Multinomial expansion for z_d^{2m} = (-1)^m (z_1^2 + ... + z_{d-1}^2)^m
            sign = Fraction((-1) ** m)
            sub_d = self.d - 1

            for partition in _partitions_of_m(sub_d, m):
                multinomial_coeff = Fraction(math.factorial(m))
                for a_i in partition:
                    multinomial_coeff //= math.factorial(a_i)

                term_coeff = coeff * sign * multinomial_coeff

                new_exp = [b + 2 * a for b, a in zip(base_exp, partition)] + [r]
                t = tuple(new_exp)
                self.terms[t] = self.terms.get(t, Fraction(0)) + term_coeff
                if self.terms[t] == 0:
                    del self.terms[t]
        else:
            t = tuple(exp_list)
            self.terms[t] = self.terms.get(t, Fraction(0)) + coeff
            if self.terms[t] == 0:
                del self.terms[t]

    def normal_form(self) -> 'QuadricQuotientPolynomial':
        """Invariant: QuadricQuotientPolynomial is always in canonical normal form."""
        return self

    def multiply_by_x(self, var_idx: int = 0) -> 'QuadricQuotientPolynomial':
        """Multiplies by variable z_{var_idx+1} (default z_1 = x) in quotient ring R(Q)."""
        if not (0 <= var_idx < self.d):
            raise ValueError(f"var_idx={var_idx} must be in range [0, {self.d - 1}]")

        res = QuadricQuotientPolynomial(self.d)
        for exp_tuple, coeff in self.terms.items():
            new_exp = list(exp_tuple)
            new_exp[var_idx] += 1
            res._add_term(tuple(new_exp), coeff)
        return res

    def evaluate(self, point: List[float], on_quadric_check: bool = False) -> float:
        """
        Evaluates the canonical normal form representative at a point in R^d.
        If on_quadric_check is True, asserts that sum(point_i^2) == 0.
        """
        if len(point) != self.d:
            raise ValueError(f"Point length must be {self.d}")

        if on_quadric_check:
            q_val = sum(pt * pt for pt in point)
            if abs(q_val) > 1e-8:
                raise ValueError(f"Point {point} is not on the quadric sum z_i^2 = 0 (q = {q_val})")

        total = 0.0
        for exp_tuple, coeff in self.terms.items():
            monomial_val = 1.0
            for var_val, exp in zip(point, exp_tuple):
                monomial_val *= (var_val ** exp)
            total += float(coeff) * monomial_val
        return total

    def isclose(self, other: 'QuadricQuotientPolynomial', atol: float = 1e-10) -> bool:
        """Floating point tolerance comparison for polynomial terms."""
        if self.d != other.d:
            return False
        all_keys = set(self.terms.keys()).union(set(other.terms.keys()))
        for k in all_keys:
            v1 = float(self.terms.get(k, 0))
            v2 = float(other.terms.get(k, 0))
            if abs(v1 - v2) > atol:
                return False
        return True

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, QuadricQuotientPolynomial):
            return False
        return self.d == other.d and self.terms == other.terms

    def __repr__(self) -> str:
        if not self.terms:
            return "0"
        str_terms = []
        for exp_tuple, coeff in sorted(self.terms.items()):
            coeff_str = str(coeff)
            monomial_parts = []
            for i, exp in enumerate(exp_tuple):
                if exp == 1:
                    monomial_parts.append(f"z{i+1}")
                elif exp > 1:
                    monomial_parts.append(f"z{i+1}^{exp}")

            monomial_str = "*".join(monomial_parts)
            if monomial_str:
                str_terms.append(f"{coeff_str}*{monomial_str}")
            else:
                str_terms.append(f"{coeff_str}")

        return " + ".join(str_terms).replace("+ -", "- ")


def _partitions_of_m(length: int, m: int) -> List[Tuple[int, ...]]:
    """Helper returning all tuples (a_1, ..., a_length) of non-negative integers with sum = m."""
    if length == 1:
        return [(m,)]
    res = []
    for i in range(m + 1):
        for sub in _partitions_of_m(length - 1, m - i):
            res.append((i,) + sub)
    return res


def pieri_coefficients(n: int, lambda_val: Union[float, Fraction], exact: bool = False) -> Tuple[Union[float, Fraction], Union[float, Fraction]]:
    """
    Computes three-term recurrence coefficients for unnormalized Gegenbauer polynomials C_n^(lambda):
      x * C_n^(lambda) = C_+ * C_{n+1}^(lambda) + C_- * C_{n-1}^(lambda)
    """
    if lambda_val <= 0:
        raise ValueError("lambda_val must be > 0 for spherical Gegenbauer functions")

    if exact:
        lam_frac = Fraction(lambda_val)
        c_plus = Fraction(n + 1, 2 * (n + lam_frac))
        c_minus = Fraction(n + 2 * lam_frac - 1, 2 * (n + lam_frac))
        return c_plus, c_minus
    else:
        lam_f = float(lambda_val)
        c_plus = (n + 1.0) / (2.0 * (n + lam_f))
        c_minus = (n + 2.0 * lam_f - 1.0) / (2.0 * (n + lam_f))
        return c_plus, c_minus


def normalized_jacobi_coefficients(n: int, lambda_val: Union[float, Fraction], exact: bool = False) -> Tuple[Union[float, Fraction], Union[float, Fraction]]:
    """
    Computes exact normalized Jacobi recurrence coefficients for zonal functions phi_n(1) = 1:
      M_x phi_n = a_n * phi_{n+1} + b_n * phi_{n-1}
    where:
      a_n = (n + 2*lambda) / (2 * (n + lambda))
      b_n = n / (2 * (n + lambda))
      a_n + b_n = 1 exactly.
    """
    if lambda_val <= 0:
        raise ValueError("lambda_val must be > 0 for spherical Gegenbauer functions")

    if exact:
        lam_frac = Fraction(lambda_val)
        a_n = Fraction(n + 2 * lam_frac, 2 * (n + lam_frac))
        b_n = Fraction(n, 2 * (n + lam_frac))
        return a_n, b_n
    else:
        lam_f = float(lambda_val)
        a_n = (n + 2.0 * lam_f) / (2.0 * (n + lam_f))
        b_n = n / (2.0 * (n + lam_f))
        return a_n, b_n


def orthonormal_jacobi_coefficients(n: int, lambda_val: float) -> float:
    """
    Computes symmetric subdiagonal coefficient alpha_n for self-adjoint Jacobi matrix J = J^*:
      M_x e_n = alpha_n e_{n+1} + alpha_{n-1} e_{n-1}
    where e_n = phi_n / ||phi_n|| is the orthonormal basis:
      alpha_n = 1/2 * sqrt( (n + 1)(n + 2*lambda) / ((n + lambda)(n + lambda + 1)) ).
    Sanity check for lambda = 0.5 (Legendre): alpha_0 = 1 / sqrt(3).
    """
    if lambda_val <= 0:
        raise ValueError("lambda_val must be > 0 for spherical Gegenbauer functions")
    num = (n + 1.0) * (n + 2.0 * lambda_val)
    den = (n + lambda_val) * (n + lambda_val + 1.0)
    return 0.5 * math.sqrt(num / den)


def pochhammer(a: Union[float, Fraction], k: int) -> Union[float, Fraction]:
    """Computes rising Pochhammer symbol (a)_k = a * (a+1) * ... * (a+k-1)."""
    if k < 0:
        raise ValueError("k must be non-negative integer")
    if isinstance(a, Fraction) or isinstance(a, int):
        val = Fraction(1)
        for i in range(k):
            val *= (a + i)
        return val
    else:
        val = 1.0
        for i in range(k):
            val *= (a + i)
        return val


def normalized_gegenbauer_2f1_coefficients(n: int, lambda_val: float) -> List[float]:
    """
    Computes exact hypergeometric expansion coefficients for normalized zonal function:
      phi_n(x) = C_n^(lambda)(x) / C_n^(lambda)(1) = sum_{k=0}^n c_k * t^k, where t = (1-x)/2.
    Formula:
      c_k = (-1)^k * binom(n, k) * (n + 2*lambda)_k / (lambda + 1/2)_k.
    """
    if lambda_val <= 0:
        raise ValueError("lambda_val must be > 0 for spherical Gegenbauer functions")

    coeffs = []
    for k in range(n + 1):
        num = ((-1.0) ** k) * math.comb(n, k) * float(pochhammer(n + 2.0 * lambda_val, k))
        den = float(pochhammer(lambda_val + 0.5, k))
        coeffs.append(num / den)
    return coeffs


# Alias for backward compatibility
schubert_intersection_coefficients = normalized_gegenbauer_2f1_coefficients


def exact_rational_gegenbauer(n: int, lambda_val: Union[int, Fraction], x: Union[int, Fraction]) -> Fraction:
    """
    Evaluates Gegenbauer polynomial C_n^(lambda)(x) exactly in Q[lambda, x]
    using the three-term recurrence (bypassing Gamma, factorials, and floating-point errors).

    C_0^(lambda)(x) = 1
    C_1^(lambda)(x) = 2 * lambda * x
    n * C_n^(lambda)(x) = 2*(n + lambda - 1)*x * C_{n-1}^(lambda)(x) - (n + 2*lambda - 2) * C_{n-2}^(lambda)(x)
    """
    if n < 0:
        raise ValueError("Degree n must be non-negative integer")
    lam = Fraction(lambda_val)
    x_frac = Fraction(x)

    if n == 0:
        return Fraction(1)
    if n == 1:
        return 2 * lam * x_frac

    c_prev = Fraction(1)
    c_curr = 2 * lam * x_frac

    for k in range(2, n + 1):
        # k * C_k = 2 * (k + lam - 1) * x * c_curr - (k + 2*lam - 2) * c_prev
        term1 = 2 * (k + lam - 1) * x_frac * c_curr
        term2 = (k + 2 * lam - 2) * c_prev
        c_next = (term1 - term2) / k
        c_prev, c_curr = c_curr, c_next

    return c_curr


def exact_rational_gegenbauer_derivative(n: int, lambda_val: Union[int, Fraction], x: Union[int, Fraction]) -> Fraction:
    """
    Computes exact first derivative d/dx C_n^(lambda)(x) = 2 * lambda * C_{n-1}^(lambda + 1)(x) in Q[lambda, x].
    """
    if n < 0:
        raise ValueError("Degree n must be non-negative integer")
    if n == 0:
        return Fraction(0)

    lam = Fraction(lambda_val)
    x_frac = Fraction(x)
    return 2 * lam * exact_rational_gegenbauer(n - 1, lam + 1, x_frac)


def exact_rational_gegenbauer_second_derivative(n: int, lambda_val: Union[int, Fraction], x: Union[int, Fraction]) -> Fraction:
    """
    Computes exact second derivative d^2/dx^2 C_n^(lambda)(x) in Q[lambda, x]:
      - For x != +-1: uses ODE substitution y'' = ((2*lambda + 1)*x*y' - n*(n + 2*lambda)*y) / (1 - x^2)
      - For x = +-1 or general: shifted formula 4 * lambda * (lambda + 1) * C_{n-2}^(lambda + 2)(x)
    """
    if n < 2:
        return Fraction(0)

    lam = Fraction(lambda_val)
    x_frac = Fraction(x)

    if x_frac**2 != 1:
        y = exact_rational_gegenbauer(n, lam, x_frac)
        y_prime = exact_rational_gegenbauer_derivative(n, lam, x_frac)
        num = (2 * lam + 1) * x_frac * y_prime - n * (n + 2 * lam) * y
        den = 1 - x_frac**2
        return num / den
    else:
        return 4 * lam * (lam + 1) * exact_rational_gegenbauer(n - 2, lam + 2, x_frac)


def exact_rational_bit_length(n: int, lambda_val: Union[int, Fraction], x: Union[int, Fraction]) -> Tuple[int, int]:
    """
    Computes numerator and denominator bit-length B_bits(n) for exact Gegenbauer polynomial C_n^(lambda)(x).
    Returns tuple (num_bit_len, den_bit_len).
    """
    val = exact_rational_gegenbauer(n, lambda_val, x)
    num_bits = abs(val.numerator).bit_length()
    den_bits = abs(val.denominator).bit_length()
    return num_bits, den_bits


def modular_gegenbauer_recurrence(n: int, lambda_val: Union[int, Fraction], x: int, mod: int) -> int:
    """
    Evaluates Gegenbauer polynomial C_n^(lambda)(x) in the modular ring Z/mod Z
    using unsigned cyclic wrapping arithmetic (hardware overflow simulation).
    Admissibility requirement: mod > n (or gcd(k, mod) == 1 for all k in [2, n])
    and if lambda = a/b, gcd(b, mod) == 1 (mod > 2, p > n, p \nmid b).
    """
    if n < 0:
        raise ValueError("n must be non-negative integer")
    if mod <= 2:
        raise ValueError("Modulus must be > 2 for standard Gegenbauer recurrence (p > 2)")

    if isinstance(lambda_val, Fraction):
        a, b = lambda_val.numerator, lambda_val.denominator
        if math.gcd(b, mod) != 1:
            raise ValueError(f"gcd(den(lambda)={b}, mod={mod}) != 1: lambda denominator not invertible mod {mod}")
        b_inv = pow(b, -1, mod)
        lam = (a * b_inv) % mod
    else:
        lam = int(lambda_val) % mod

    x_mod = x % mod

    if n == 0:
        return 1 % mod
    if n == 1:
        return (2 * lam * x_mod) % mod

    c_prev = 1 % mod
    c_curr = (2 * lam * x_mod) % mod

    for k in range(2, n + 1):
        if math.gcd(k, mod) != 1:
            raise ValueError(f"gcd(k={k}, mod={mod}) != 1: degree k={k} not invertible mod {mod} (requires mod > n)")
        k_inv = pow(k, -1, mod)
        term1 = (2 * (k + lam - 1) * x_mod * c_curr) % mod
        term2 = ((k + 2 * lam - 2) * c_prev) % mod
        c_next = ((term1 - term2) * k_inv) % mod
        c_prev, c_curr = c_curr, c_next

    return c_curr


def rns_crt_gegenbauer_eval(n: int, lambda_val: int, x: int, moduli: List[int]) -> int:
    """
    Evaluates high-degree integer Gegenbauer polynomial C_n^(lambda)(x) using
    Residue Number System (RNS) over pairwise coprime word-size moduli and reconstructs
    the exact full-precision integer via Chinese Remainder Theorem (CRT), subject to
    a-priori magnitude bound |X| < M / 2 where M = prod(moduli).
    """
    residues = []
    for m in moduli:
        r = modular_gegenbauer_recurrence(n, lambda_val, x, m)
        residues.append(r)

    # Chinese Remainder Theorem Reconstruction
    M = 1
    for m in moduli:
        M *= m

    X = 0
    for r_i, m_i in zip(residues, moduli):
        M_i = M // m_i
        y_i = pow(M_i, -1, m_i)
        X = (X + r_i * M_i * y_i) % M

    # Signed integer conversion
    if X > M // 2:
        X -= M

    return X


def gauss_gegenbauer_quadrature(n: int, lambda_val: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes Gauss-Gegenbauer quadrature nodes x_k and weights w_k using Golub-Welsch
    spectral decomposition of the symmetric tridiagonal Jacobi matrix J_n.
    Golub-Welsch isolates the algebraic spectral data (x_k, v_{k,1}^2) from the global
    transcendental scalar normalization mu_0 = sqrt(pi) * gamma(lambda + 0.5) / gamma(lambda + 1.0).
    Avoids direct endpoint evaluation at x = +-1, reducing endpoint singularity exposure.
    Weight function: w(x) = (1-x^2)^(lambda - 1/2) over [-1, 1].
    """
    if n <= 0:
        raise ValueError("Number of quadrature nodes n must be > 0")
    if lambda_val <= -0.5:
        raise ValueError("lambda_val must be > -0.5")

    # Build symmetric tridiagonal Jacobi matrix J_n using orthonormal subdiagonal alpha_k
    subdiag = np.zeros(n - 1, dtype=np.float64)
    for k in range(n - 1):
        subdiag[k] = orthonormal_jacobi_coefficients(k, lambda_val)

    J = np.diag(subdiag, k=1) + np.diag(subdiag, k=-1)

    # Golub-Welsch algorithm: Eigendecomposition of symmetric tridiagonal J
    nodes, eigenvectors = np.linalg.eigh(J)

    # Total integral weight norm mu_0
    mu_0 = float(math.sqrt(math.pi) * gamma(lambda_val + 0.5) / gamma(lambda_val + 1.0))

    # Weights w_k = mu_0 * (v_{k, 0})^2 where v_{k, 0} is first component of k-th normalized eigenvector
    weights = mu_0 * (eigenvectors[0, :] ** 2)

    return nodes, weights


if __name__ == "__main__":
    print("--- REFACTORED ALGEBRAIC GEOMETRY & COMBINATORICS DEMO ---")
    d_dim = 5  # d=5 -> Lambda=1.5, Q_3 c P^4
    n_deg = 4
    lam_frac = lambda_for_sphere(d_dim)

    h0 = quadric_hilbert_series_dim(d_dim, n_deg)
    a_n, b_n = normalized_jacobi_coefficients(n_deg, lam_frac, exact=True)
    alpha_0_legendre = orthonormal_jacobi_coefficients(0, 0.5)

    print(f"Projective Quadric Q_{d_dim-2} c P^{d_dim-1}, degree n={n_deg}:")
    print(f"  * Sphere Gegenbauer Parameter Lambda: {lam_frac}")
    print(f"  * Hilbert Series Dimension dim R(Q)_{n_deg}: {h0}")
    print(f"  * Exact Rational Jacobi Recurrence Coefficients (a_n, b_n): ({a_n}, {b_n}) [a_n + b_n = {a_n + b_n}]")
    print(f"  * Orthonormal Symmetric Jacobi Matrix Subdiagonal alpha_0 (lambda=0.5): {alpha_0_legendre:.6f} [expected 1/sqrt(3) = {1/math.sqrt(3):.6f}]")

    # Test Quotient Ring Normal Form
    poly = QuadricQuotientPolynomial(3, {(0, 0, 2): 1})  # z_3^2 in C[z_1, z_2, z_3]/(z_1^2+z_2^2+z_3^2)
    print(f"  * Exact Normal Form of z_3^2 modulo q: {poly}")
