"""
Algebraic Geometry and Combinatorics of Projective Quadrics
============================================================
This module implements exact algebraic geometry and combinatorial operations
for Gegenbauer polynomials and complex projective quadric hypersurfaces Q_{d-2} c P^{d-1}:
1. Hilbert Polynomial h^0(Q_{d-2}, O(n)) via Hilbert series H_{R(Q)}(t) = (1-t^2)/(1-t)^d
   with binomial coefficient zeroing convention binom(r, d-1) = 0 for r < d-1 (or Sym^m(C^d) = 0 for m < 0).
2. Exact Rational Quotient Ring Normal Forms: Polynomial remainder modulo q = sum(z_i^2) in C[z_1,...,z_d]/(q)
   Note: Normal form reduction convention maps z_d^2 -> -(z_1^2 + ... + z_{d-1}^2)
3. Exact Rational Polynomial Path C_n^(lambda)(x) in Q[lambda, x] vs Evaluation Theorem: lambda, x in Q => C_n^(lambda)(x) in Q.
4. Orthonormal Symmetric Jacobi Matrix Coefficients alpha_n = 1/2 * sqrt( (n+1)(n+2*lambda) / ((n+lambda)(n+lambda+1)) )
5. Golub-Welsch Gauss-Gegenbauer Quadrature over m x m principal truncation J_m (sigma(J_m) = {x_1, ..., x_m})
6. Normalized Gegenbauer _2F_1 Hypergeometric Expansion Coefficients
7. EndpointClass Enum, EndpointBoundaryCondition, and Physical vs Analytic Continuation Parameter Domain Classifier
8. Execution Metadata N_max & Split Algorithm/Point Denominators D_rec and D_eval
9. PolyCertificate, PointCertificate, PolyEvaluationCertificate, and ZonalCertificate Admissibility
   with 3 distinct bad-prime failure modes:
   - PointLocalizationFailure (p | r)
   - NormalizationRepresentationFailure (p | v_n)
   - NormalizationSingularityFailure (p | u_n)
"""

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
import math
from typing import Dict, List, Tuple, Union

import numpy as np

Number = Union[int, Fraction, float]


class EndpointClass(Enum):
    """Layer III Sturm-Liouville Endpoint Classification Enum."""
    CRITICAL_LC = "CRITICAL_LIMIT_CIRCLE"
    REGULAR = "REGULAR_ENDPOINT"
    LIMIT_CIRCLE = "LIMIT_CIRCLE"
    LIMIT_POINT = "LIMIT_POINT"


@dataclass
class EndpointBoundaryCondition:
    """Represents explicit two-sided endpoint boundary conditions and indicial root metadata."""
    left: EndpointClass
    right: EndpointClass
    indicial_root_plus: float = 0.5
    indicial_root_minus: float = 0.5
    forbidden_log_coeff_zero: bool = True


def physical_classifier(lambda_val: Union[float, Fraction]) -> EndpointClass:
    """Classifies parameters within PhysicalSphereDomain (d >= 3 => lambda in {1/2, 1, 3/2, ...})."""
    lam = float(lambda_val)
    if abs(lam - 0.5) < 1e-12:
        return EndpointClass.CRITICAL_LC
    elif abs(lam - 1.0) < 1e-12:
        return EndpointClass.REGULAR
    elif lam >= 1.5:
        return EndpointClass.LIMIT_POINT
    return EndpointClass.LIMIT_CIRCLE


def analytic_classifier(lambda_val: Union[float, Fraction]) -> EndpointClass:
    """Classifies parameters within AnalyticContinuationDomain (lambda > 0)."""
    lam = float(lambda_val)
    if abs(lam - 0.5) < 1e-12:
        return EndpointClass.CRITICAL_LC
    elif abs(lam - 1.0) < 1e-12:
        return EndpointClass.REGULAR
    elif lam < 1.5:
        return EndpointClass.LIMIT_CIRCLE
    else:
        return EndpointClass.LIMIT_POINT


def classify_parameter_domain(lambda_val: Union[float, Fraction], is_physical_domain: bool = True) -> EndpointClass:
    """
    Explicit Classifier Precedence Function:
      Classify(lambda) = PhysicalClassifier(lambda) if lambda in PhysicalSphereDomain else AnalyticClassifier(lambda)
    """
    if is_physical_domain:
        return physical_classifier(lambda_val)
    else:
        return analytic_classifier(lambda_val)


# Alias endpoint_class for backwards compatibility
endpoint_class = classify_parameter_domain


def get_endpoint_boundary_condition(lambda_val: Union[float, Fraction]) -> EndpointBoundaryCondition:
    """Returns explicit two-sided EndpointBoundaryCondition with indicial root metadata."""
    cls = endpoint_class(lambda_val)
    lam = float(lambda_val)
    r_plus = max(lam, 1.0 - lam) if lam < 1.5 else lam
    r_minus = min(lam, 1.0 - lam) if lam < 1.5 else 1.0 - lam
    return EndpointBoundaryCondition(
        left=cls,
        right=cls,
        indicial_root_plus=r_plus,
        indicial_root_minus=r_minus,
        forbidden_log_coeff_zero=True
    )


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
    With binomial zeroing convention binom(r, d-1) = 0 for r < d-1, this formula is uniformly valid for all n >= 0.
    """
    if d < 3 or n < 0:
        raise ValueError("Ambient Euclidean dimension d must be >= 3 and degree n >= 0.")

    term1 = math.comb(n + d - 1, d - 1) if (n + d - 1) >= (d - 1) else 0
    term2 = math.comb(n + d - 3, d - 1) if (n + d - 3) >= (d - 1) else 0
    return term1 - term2


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
                multinomial_coeff = math.factorial(m)
                for a_i in partition:
                    multinomial_coeff //= math.factorial(a_i)

                term_coeff = coeff * sign * Fraction(multinomial_coeff)

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


def phi_norm_squared(n: int, lambda_val: float) -> float:
    """
    Computes exact closed-form L^2 norm squared ||phi_n||_lambda^2 for normalized zonal function:
      ||phi_n||_lambda^2 = (pi * 2^{1-2*lambda} * n! * [Gamma(2*lambda)]^2) / ((n+lambda) * [Gamma(lambda)]^2 * Gamma(n+2*lambda)).
    Evaluated stably in log-gamma domain to avoid factorial/gamma numerical overflow for n >= 171.
    Uniformly valid for all n >= 0.
    """
    log_num = (
        math.log(math.pi)
        + (1.0 - 2.0 * lambda_val) * math.log(2.0)
        + math.lgamma(n + 1)
        + 2.0 * math.lgamma(2.0 * lambda_val)
    )
    log_den = (
        math.log(n + lambda_val)
        + 2.0 * math.lgamma(lambda_val)
        + math.lgamma(n + 2.0 * lambda_val)
    )
    log_val = log_num - log_den

    return math.exp(log_val)


def dual_recurrence_conversion(a_n: float, b_n: float, h_n: float, h_np1: float, h_nm1: float) -> Tuple[float, float]:
    """
    Computes exact commutative conversion square between polynomial-normalized recurrence (a_n, b_n)
    and orthonormal Jacobi recurrence (alpha_n, alpha_{n-1}) via norm weights h_n = ||phi_n||^{-1}:
      alpha_n = a_n * (h_n / h_{n+1})
      alpha_{n-1} = b_n * (h_n / h_{n-1})
    """
    alpha_n = a_n * (h_n / h_np1)
    alpha_nm1 = b_n * (h_n / h_nm1)
    return alpha_n, alpha_nm1


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
    Symbolic Polynomial Identity: C_n^(lambda)(x) in Q[lambda, x].
    Evaluation Theorem: lambda, x in Q => C_n^(lambda)(x) in Q.
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
    Computes exact second derivative d^2/dx^2 C_n^(lambda)(x) in Q[lambda, x]
    using universal shifted identity d^2/dx^2 C_n^(lambda)(x) = 4 * lambda * (lambda + 1) * C_{n-2}^(lambda + 2)(x).
    """
    if n < 2:
        return Fraction(0)

    lam = Fraction(lambda_val)
    x_frac = Fraction(x)
    return 4 * lam * (lam + 1) * exact_rational_gegenbauer(n - 2, lam + 2, x_frac)


def exact_rational_bit_length(n: int, lambda_val: Union[int, Fraction], x: Union[int, Fraction]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Computes numerator and denominator bit-lengths B_bits^(C)(n) for C_n^(lambda)(x)
    and B_bits^(phi)(n) for normalized spherical function phi_n(x).
    Returns tuple ((num_bits_C, den_bits_C), (num_bits_phi, den_bits_phi)).
    """
    c_val = exact_rational_gegenbauer(n, lambda_val, x)
    c1_val = exact_rational_gegenbauer(n, lambda_val, Fraction(1))
    phi_val = c_val / c1_val if c1_val != 0 else Fraction(0)

    bits_C = (abs(c_val.numerator).bit_length(), abs(c_val.denominator).bit_length())
    bits_phi = (abs(phi_val.numerator).bit_length(), abs(phi_val.denominator).bit_length())
    return bits_C, bits_phi


def extract_plan_recurrence_denominator(execution_plan: object = None, degree: int = 0, lambda_val: Union[int, Fraction] = Fraction(1, 2)) -> int:
    """
    Computes D_rec(P) = lcm({den_red(q) : q in P_recurrence_arithmetic}).
    Plan-pure recurrence denominator extraction without extra b factor.
    If execution_plan is an object with 'recurrence_arithmetic', extracts denominators from it.
    Otherwise computes D_rec directly for degree n and parameter lambda.
    """
    if execution_plan is not None and hasattr(execution_plan, 'recurrence_arithmetic'):
        lcm_val = 1
        for q in getattr(execution_plan, 'recurrence_arithmetic'):
            lcm_val = math.lcm(lcm_val, Fraction(q).denominator)
        return lcm_val

    lcm_val = 1
    for k in range(1, degree + 1):
        a_k, b_k = normalized_jacobi_coefficients(k, Fraction(lambda_val), exact=True)
        lcm_val = math.lcm(lcm_val, Fraction(a_k).denominator)
        lcm_val = math.lcm(lcm_val, Fraction(b_k).denominator)
    return lcm_val


def get_recurrence_denominator_lcm(n: int, lambda_val: Union[int, Fraction]) -> int:
    """
    Computes D_rec = lcm({den_red(a_k), den_red(b_k) : k = 1..n} u {b}) for parameter lambda = a/b.
    """
    return extract_plan_recurrence_denominator(execution_plan=None, degree=n, lambda_val=lambda_val)


# Alias for backward compatibility
get_algorithm_denominator_lcm = get_recurrence_denominator_lcm


def get_normalization_denominator(n: int, lambda_val: Union[int, Fraction]) -> int:
    """
    Computes D_norm = den(C_n^(lambda)(1)) * den(||phi_n||_lambda^2).
    """
    c1 = exact_rational_gegenbauer(n, lambda_val, Fraction(1))
    v_n = c1.denominator
    # den(h_n^2) = den(||phi_n||_lambda^2)
    norm_sq = phi_norm_squared(n, float(lambda_val))
    norm_frac = Fraction(norm_sq).limit_denominator(1000000)
    return v_n * norm_frac.denominator


def get_evaluation_denominator(x: Union[int, Fraction]) -> int:
    """Computes D_eval = r for evaluation point x = c/r."""
    return Fraction(x).denominator


def get_excluded_primes_denominator(n: int, lambda_val: Union[int, Fraction], x: Union[int, Fraction]) -> int:
    """Computes D_excl = lcm(D_rec, D_norm, D_eval)."""
    d_rec = get_recurrence_denominator_lcm(n, lambda_val)
    d_norm = get_normalization_denominator(n, lambda_val)
    d_eval = get_evaluation_denominator(x)
    return math.lcm(d_rec, math.lcm(d_norm, d_eval))


def check_c_n_admissibility(n: int, lambda_val: Union[int, Fraction], p: int, execution_plan: object = None) -> bool:
    """
    Admissibility certificate for unnormalized Gegenbauer polynomial C_n^(lambda)(x) in F_p (PolyCertificate):
      PolyCertificate(p, P, n) requires Prime(p) and gcd(p, D_rec(P)) == 1.
    """
    if p <= 1:
        return False
    # Check primality for F_p field structure
    for i in range(2, int(math.isqrt(p)) + 1):
        if p % i == 0:
            return False

    d_rec = extract_plan_recurrence_denominator(execution_plan=execution_plan, degree=n, lambda_val=lambda_val)
    return (d_rec % p != 0)


def zonal_admissible(p: int, degree: int, point: Union[int, Fraction], lambda_val: Union[int, Fraction] = Fraction(1, 2), execution_plan: object = None) -> bool:
    """
    Semantically exact ZonalAdmissible predicate over canonical reduced fractions
      C_n^(lambda)(1) = u_n / v_n with gcd(u_n, v_n) = 1, v_n > 0
      and x = c / r with gcd(c, r) = 1, r > 0:
      ZonalAdmissible(p, P, n, x) <=> Prime(p) and gcd(p, D_rec(P)) == 1 and p nmid r and p nmid u_n * v_n.
    """
    if p <= 1:
        return False
    for i in range(2, int(math.isqrt(p)) + 1):
        if p % i == 0:
            return False

    r_frac = Fraction(point)
    c, r = r_frac.numerator, r_frac.denominator
    if r % p == 0:
        return False

    d_rec = extract_plan_recurrence_denominator(execution_plan=execution_plan, degree=degree, lambda_val=lambda_val)
    if d_rec % p != 0: # wait, if d_rec % p == 0 then p | d_rec, not admissible
        pass
    if d_rec % p == 0:
        return False

    c1 = exact_rational_gegenbauer(degree, lambda_val, Fraction(1))
    u_n = abs(c1.numerator)
    v_n = c1.denominator

    if v_n % p == 0 or u_n % p == 0:
        return False

    return True


def bad_zonal_prime(p: int, degree: int, point: Union[int, Fraction], lambda_val: Union[int, Fraction] = Fraction(1, 2), execution_plan: object = None) -> bool:
    """
    Semantically exact bad zonal prime predicate defined as the logical negation:
      Bad_zonal := not ZonalAdmissible.
    """
    return not zonal_admissible(p=p, degree=degree, point=point, lambda_val=lambda_val, execution_plan=execution_plan)


def check_point_admissibility(x: Union[int, Fraction], p: int) -> bool:
    """
    Admissibility certificate for evaluation point x = c/r in F_p (PointCertificate):
      Requires gcd(p, r) == 1.
    """
    r = Fraction(x).denominator
    return (r % p != 0)


def check_poly_evaluation_admissibility(n: int, lambda_val: Union[int, Fraction], x: Union[int, Fraction], p: int) -> bool:
    """
    Admissibility certificate for polynomial evaluation C_n^(lambda)(x) in F_p (PolyEvaluationCertificate):
      PolyEvaluationCertificate = PolyCertificate and PointCertificate.
    """
    return check_c_n_admissibility(n, lambda_val, p) and check_point_admissibility(x, p)


def check_zonal_admissibility(p: int, execution_plan: object, degree: int, point: Union[int, Fraction], lambda_val: Union[int, Fraction] = Fraction(1, 2)) -> Tuple[bool, str]:
    """
    Admissibility certificate for normalized zonal spherical function phi_n(x) = C_n^(lambda)(x) / C_n^(lambda)(1) in F_p:
      ZonalCertificate(p, execution_plan, degree=n, point=x=c/r)
      Formal logical dependency: ZonalCertificate = PolyEvaluationCertificate and (p nmid v_n) and (C_n(1) != 0 mod p)
      Failure Mode Taxonomy:
        1. PointLocalizationFailure (p | r or p divides D_rec): non-local in F_p.
        2. NormalizationRepresentationFailure (p | v_n): rational representation u_n/v_n of C_n(1) non-local in F_p (p | v_n).
        3. NormalizationSingularityFailure (p | u_n): C_n(1) == 0 mod p, normalization division by zero in F_p (p | u_n).
    """
    n = degree
    x = point

    if not check_c_n_admissibility(n, lambda_val, p):
        return False, "PointLocalizationFailure: p non-prime or divides recurrence denominator product D_rec"

    if not check_point_admissibility(x, p):
        return False, "PointLocalizationFailure: p divides evaluation point denominator r"

    c1 = exact_rational_gegenbauer(n, lambda_val, Fraction(1))
    u_n = c1.numerator
    v_n = c1.denominator

    if v_n % p == 0:
        return False, f"NormalizationRepresentationFailure: p={p} divides C_n(1) denominator v_n={v_n}"

    if u_n % p == 0:
        return False, f"NormalizationSingularityFailure: bad prime p={p} divides C_n(1) numerator u_n={u_n} (C_n(1) == 0 mod p)"

    return True, "Valid ZonalCertificate"


# Alias for backward compatibility
def check_phi_n_admissibility(n: int, lambda_val: Union[int, Fraction], x: Union[int, Fraction], p: int) -> Tuple[bool, str]:
    return check_zonal_admissibility(p=p, execution_plan=None, degree=n, point=x, lambda_val=lambda_val)


def modular_gegenbauer_recurrence(n: int, lambda_val: Union[int, Fraction], x: Union[int, Fraction], mod: int) -> int:
    """
    Evaluates Gegenbauer polynomial C_n^(lambda)(x) in the modular ring Z/mod Z.
    Characteristic p admissibility requires p > max(2, n), p \nmid b (for lambda = a/b),
    and p \nmid r (for x = c/r).
    """
    if n < 0:
        raise ValueError("n must be non-negative integer")
    if mod <= max(2, n):
        raise ValueError(f"Modulus {mod} must be > max(2, n={n}) for characteristic p > n recurrence evaluation")

    if isinstance(lambda_val, Fraction):
        a, b = lambda_val.numerator, lambda_val.denominator
        if math.gcd(b, mod) != 1:
            raise ValueError(f"gcd(den(lambda)={b}, mod={mod}) != 1: lambda denominator not invertible mod {mod}")
        b_inv = pow(b, -1, mod)
        lam = (a * b_inv) % mod
    else:
        lam = int(lambda_val) % mod

    if isinstance(x, Fraction):
        c, r = x.numerator, x.denominator
        if math.gcd(r, mod) != 1:
            raise ValueError(f"gcd(den(x)={r}, mod={mod}) != 1: x denominator not invertible mod {mod}")
        r_inv = pow(r, -1, mod)
        x_mod = (c * r_inv) % mod
    else:
        x_mod = int(x) % mod

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


def rns_crt_gegenbauer_eval(n: int, lambda_val: Union[int, Fraction], x: Union[int, Fraction], moduli: List[int]) -> int:
    """
    Evaluates high-degree integer Gegenbauer polynomial C_n^(lambda)(x) using
    Residue Number System (RNS) over pairwise coprime word-size moduli and reconstructs
    the exact full-precision integer via Chinese Remainder Theorem (CRT), subject to
    a-priori magnitude bound |X| < M / 2 where M = prod(moduli).
    """
    residues = []
    for m in moduli:
        r_mod = modular_gegenbauer_recurrence(n, lambda_val, x, m)
        residues.append(r_mod)

    # Chinese Remainder Theorem Reconstruction
    M = 1
    for m in moduli:
        M *= m

    X = 0
    for r_i, m_i in zip(residues, moduli):
        M_i = M // m_i
        y_i = pow(M_i, -1, m_i)
        X = (X + r_i * M_i * y_i) % M

    # Signed integer conversion for |X| < M / 2
    if X > M // 2:
        X -= M

    return X


def gauss_gegenbauer_quadrature(m: int, lambda_val: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes Gauss-Gegenbauer m-point quadrature nodes x_k and weights w_k using Golub-Welsch
    spectral decomposition of the symmetric m x m principal truncation Jacobi matrix J_m
    with eigenvalues sigma(J_m) = {x_1, ..., x_m}.
    Reserving n for Gegenbauer degree and m for quadrature order.
    Golub-Welsch isolates the algebraic spectral data (x_k, v_{k,1}^2) from the global
    transcendental scalar normalization mu_0 = sqrt(pi) * exp(lgamma(lambda + 0.5) - lgamma(lambda + 1.0)).
    Avoids direct endpoint evaluation at x = +-1, reducing endpoint singularity exposure.
    Weight function: w(x) = (1-x^2)^(lambda - 1/2) over [-1, 1].
    """
    if m <= 0:
        raise ValueError("Number of quadrature nodes m must be > 0")
    if lambda_val <= -0.5:
        raise ValueError("lambda_val must be > -0.5")

    # Build symmetric tridiagonal Jacobi matrix J_m using orthonormal subdiagonal alpha_k
    subdiag = np.zeros(m - 1, dtype=np.float64)
    for k in range(m - 1):
        subdiag[k] = orthonormal_jacobi_coefficients(k, lambda_val)

    J_m = np.diag(subdiag, k=1) + np.diag(subdiag, k=-1)

    # Golub-Welsch algorithm: Eigendecomposition of symmetric tridiagonal J_m
    nodes, eigenvectors = np.linalg.eigh(J_m)

    # Total integral weight norm mu_0 evaluated safely in log-gamma domain: mu_0 = B(1/2, lambda + 1/2)
    mu_0 = float(math.sqrt(math.pi) * math.exp(math.lgamma(lambda_val + 0.5) - math.lgamma(lambda_val + 1.0)))

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

    # Test Bad Prime Check
    valid_c = check_c_n_admissibility(5, Fraction(3, 2), 7)
    valid_phi, msg = check_phi_n_admissibility(5, Fraction(3, 2), Fraction(1, 2), 7)
    print(f"  * Admissibility p=7: C_5={valid_c}, phi_5={valid_phi} ({msg})")
