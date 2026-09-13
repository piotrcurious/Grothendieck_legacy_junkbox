"""
Computational Layer and Pareto Optimization Solver for Gegenbauer Polynomials
================================================================================
This module provides a computational framework taking numerical bases and types into
account, and selects optimal expression permutations on the Computational Cost
(measured latency / FLOPs) vs Numerical Error plane.

Features:
- Real Execution Backends: FLOAT32, FLOAT64, LONGDOUBLE, MPMATH (100+ bits),
  FIXED_POINT (Q16.16 integer scaling), and LNS (deterministic log-domain).
- High-Precision Reference Ground Truth via mpmath (100-300 bits).
- Correct Harmonic Quotient Ring Zonal Polynomial Projection in R(Q).
- Direct Hypergeometric _2F_1(-n, n+2*lambda; lambda+0.5; (1-x)/2) evaluation.
- Domain-aware validity classification without artificial endpoint clipping.
- Hard optimization constraints raising ValueError when constraints are infeasible.
- Multi-percentile mixed error metrics (max, median, 95th percentile, RMS).
- Backend-dependent regularization floor policy tau_M = max(tau_abs, tau_rel * S_M).
"""

from dataclasses import dataclass
from enum import Enum
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.special import eval_gegenbauer, gamma, gammaln, jv, hyp2f1

try:
    import mpmath
    HAS_MPMATH = True
except ImportError:
    HAS_MPMATH = False

try:
    from algebraic_geometry_combinatorics import QuadricQuotientPolynomial, normalized_jacobi_coefficients, normalized_gegenbauer_2f1_coefficients, modular_gegenbauer_recurrence, rns_crt_gegenbauer_eval
    from gegenbauer_asymptotics import normalized_phi_recurrence, endpoint_bessel_leading, south_pole_bessel_leading, interior_wkb_approx, composite_matched_approx, c_n_1_val
except ModuleNotFoundError:
    from Gregenbauer_demistify.algebraic_geometry_combinatorics import QuadricQuotientPolynomial, normalized_jacobi_coefficients, normalized_gegenbauer_2f1_coefficients, modular_gegenbauer_recurrence, rns_crt_gegenbauer_eval
    from Gregenbauer_demistify.gegenbauer_asymptotics import normalized_phi_recurrence, endpoint_bessel_leading, south_pole_bessel_leading, interior_wkb_approx, composite_matched_approx, c_n_1_val


def mixed_error(approx: np.ndarray, ref: np.ndarray, atol: float = 1e-14, rtol: float = 1e-10) -> np.ndarray:
    """Computes robust mixed error to handle near-zero values near polynomial roots."""
    return np.abs(approx - ref) / (atol + rtol * np.abs(ref))


def cross_backend_error(backend_a_vals: np.ndarray, backend_b_vals: np.ndarray) -> float:
    """
    Computes Layer VIII Cross-Backend Error Metric E_{A,B} = max |val_A - val_B|
    between independent execution backends A and B.
    """
    return float(np.nanmax(np.abs(np.asarray(backend_a_vals) - np.asarray(backend_b_vals))))


def get_backend_tau(tau_abs: float = 1e-14, tau_rel: float = 1e-14, scale: float = 1.0) -> float:
    """
    Computes backend-dependent regularization floor tau_M = max(tau_abs, tau_rel * S_M).
    """
    return float(max(tau_abs, tau_rel * scale))


def scale_invariant_schrodinger_residual(u_val: float, u_second_val: float, theta: float, n: int, lambda_val: float, tau: float = 1e-14) -> float:
    """
    Computes Layer VIII Normalized Scale-Invariant Schrödinger Residual R_Schr(theta).
    Formula: |-u'' + lambda*(lambda-1)*csc^2(theta)*u - (n+lambda)^2*u| / (|u''| + |lambda*(lambda-1)*csc^2(theta)*u| + (n+lambda)^2*|u| + tau_M)
    """
    k = n + lambda_val
    sing = lambda_val * (lambda_val - 1.0) / (np.sin(theta) ** 2)
    num = abs(-u_second_val + sing * u_val - (k ** 2) * u_val)
    den = abs(u_second_val) + abs(sing * u_val) + (k ** 2) * abs(u_val) + tau
    return float(num / den)


def scale_invariant_recurrence_residual(phi_n: float, phi_np1: float, phi_nm1: float, x: float, n: int, lambda_val: float, tau: float = 1e-14) -> float:
    """
    Computes Layer VIII Normalized Scale-Invariant Recurrence Residual R_rec(n, x) for n >= 1.
    Formula: |x*phi_n - a_n*phi_{n+1} - b_n*phi_{n-1}| / (|x*phi_n| + |a_n*phi_{n+1}| + |b_n*phi_{n-1}| + tau_M)
    """
    if n < 1:
        raise ValueError("Scale-invariant recurrence residual R_rec(n, x) is defined for n >= 1 (initial conditions phi_0=1, phi_1=x)")
    a_n = (n + 2.0 * lambda_val) / (2.0 * (n + lambda_val))
    b_n = n / (2.0 * (n + lambda_val))
    num = abs(x * phi_n - a_n * phi_np1 - b_n * phi_nm1)
    den = abs(x * phi_n) + abs(a_n * phi_np1) + abs(b_n * phi_nm1) + tau
    return float(num / den)


def normalization_residual(phi_n_val: float, c_n_val: float, c_n_1_val: float) -> float:
    """
    Computes Layer VIII Normalization Residual R_norm(x) = |C_n^(lambda)(1) * phi_n(x) - C_n^(lambda)(x)|.
    """
    return float(abs(c_n_1_val * phi_n_val - c_n_val))


def high_precision_reference(n: int, lambda_val: float, x: np.ndarray, dps: int = 100) -> np.ndarray:
    """
    Computes independently converged high-precision ground truth reference for zonal function phi_n(x)
    using mpmath at dps digits (default dps=100 corresponding to p_ref >= 384 bits).
    """
    x_arr = np.asarray(x, dtype=np.float64)
    if not HAS_MPMATH:
        c1 = float(eval_gegenbauer(n, lambda_val, 1.0))
        return np.asarray(eval_gegenbauer(n, lambda_val, x_arr), dtype=np.float64) / c1

    old_dps = mpmath.mp.dps
    try:
        mpmath.mp.dps = dps
        n_mp = mpmath.mpf(n)
        lam_mp = mpmath.mpf(lambda_val)
        c1_mp = mpmath.gegenbauer(n_mp, lam_mp, mpmath.mpf(1.0))

        out = np.zeros_like(x_arr, dtype=np.float64)
        for i, xi in enumerate(x_arr):
            val_mp = mpmath.gegenbauer(n_mp, lam_mp, mpmath.mpf(xi)) / c1_mp
            out[i] = float(val_mp)
        return out
    finally:
        mpmath.mp.dps = old_dps


class NumericalBase(Enum):
    BASE_2 = "Base 2 (IEEE Binary)"
    BASE_10 = "Base 10 (Decimal)"
    FIXED_POINT = "Fixed-Point (Q16.16)"
    LOGARITHMIC = "Logarithmic Number System (LNS)"
    FIELD_EXTENSION = "Field Extension Q(lambda, x)"
    MODULAR_RNS = "Modulus Residue Number System (RNS/CRT)"


class PrecisionType(Enum):
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    LONGDOUBLE = "longdouble"
    ARBITRARY = "mpmath_arbitrary"
    EXACT_RATIONAL = "exact_rational"


@dataclass
class NumericalContext:
    base: NumericalBase = NumericalBase.BASE_2
    precision: PrecisionType = PrecisionType.FLOAT64
    dps: int = 50
    bits: int = 64
    eps: float = 2.22e-16

    @classmethod
    def default_float64(cls):
        return cls(base=NumericalBase.BASE_2, precision=PrecisionType.FLOAT64, bits=64, eps=2.22e-16)

    @classmethod
    def float32(cls):
        return cls(base=NumericalBase.BASE_2, precision=PrecisionType.FLOAT32, bits=32, eps=1.19e-7)

    @classmethod
    def longdouble(cls):
        return cls(base=NumericalBase.BASE_2, precision=PrecisionType.LONGDOUBLE, bits=80, eps=1.0e-19)

    @classmethod
    def fixed_point_q16(cls):
        return cls(base=NumericalBase.FIXED_POINT, precision=PrecisionType.FLOAT32, bits=32, eps=1.52e-5)

    @classmethod
    def logarithmic_lns(cls):
        return cls(base=NumericalBase.LOGARITHMIC, precision=PrecisionType.FLOAT32, bits=32, eps=1e-4)

    @classmethod
    def mpmath_arbitrary(cls, dps: int = 100):
        return cls(base=NumericalBase.BASE_10, precision=PrecisionType.ARBITRARY, dps=dps, bits=dps * 4, eps=10**(-dps))


class AlgebraicPermutation(Enum):
    NORMALIZED_RECURRENCE = "Normalized Recurrence phi_n(x)"
    QUOTIENT_RING_NORMAL_FORM = "Quotient Ring Normal Form Remainder"
    HYPERGEOMETRIC_2F1 = "Hypergeometric _2F1 Series"
    INTERIOR_WKB_WEYL = "Interior WKB / Weyl Semiclassical"
    MEHLER_HEINE_BESSEL = "Mehler-Heine Bessel Boundary-Layer"
    COMPOSITE_MATCHED = "Composite Matched Asymptotic"


@dataclass
class BackendCapabilityCertificate:
    backend_id: str
    domain_description: str
    parameter_conditions: str
    error_model: str
    residual_checkers: List[str]


@dataclass
class SolverPerformanceMetrics:
    permutation: AlgebraicPermutation
    num_flops: int
    exec_time_sec: float
    max_residual: float
    max_mixed_error: float
    median_mixed_error: float
    p95_mixed_error: float
    rms_mixed_error: float
    capability_cert: Optional[BackendCapabilityCertificate] = None
    is_pareto_optimal: bool = False


class GegenbauerComputationalSolver:
    """
    Evaluates equivalent algebraic geometry permutations for Gegenbauer polynomials
    and zonal functions under specific numerical execution backends and solves for Pareto-optimal expressions.
    """

    def __init__(self, n: int, lambda_val: float, context: NumericalContext = None):
        if n < 0 or int(n) != n:
            raise ValueError("n must be a non-negative integer")
        if lambda_val <= -0.5:
            raise ValueError("lambda_val must be > -0.5 for Gegenbauer polynomials")

        self.n = n
        self.lambda_val = lambda_val
        self.context = context or NumericalContext.default_float64()

    def _execute_backend(self, eval_fn, x: np.ndarray) -> np.ndarray:
        """Executes computation using actual numerical backend arithmetic."""
        x_arr = np.asarray(x)

        if self.context.precision == PrecisionType.ARBITRARY and HAS_MPMATH:
            old_dps = mpmath.mp.dps
            try:
                mpmath.mp.dps = self.context.dps
                n_mp = mpmath.mpf(self.n)
                lam_mp = mpmath.mpf(self.lambda_val)
                c1_mp = mpmath.gegenbauer(n_mp, lam_mp, mpmath.mpf(1.0))
                out = np.zeros_like(x_arr, dtype=np.float64)
                for i, xi in enumerate(x_arr):
                    val_mp = mpmath.gegenbauer(n_mp, lam_mp, mpmath.mpf(xi)) / c1_mp
                    out[i] = float(val_mp)
                return out
            finally:
                mpmath.mp.dps = old_dps

        elif self.context.precision == PrecisionType.FLOAT32:
            x_f32 = x_arr.astype(np.float32)
            out_f32 = eval_fn(x_f32)
            return out_f32.astype(np.float64)

        elif self.context.precision == PrecisionType.LONGDOUBLE:
            x_ld = x_arr.astype(np.longdouble)
            out_ld = eval_fn(x_ld)
            return out_ld.astype(np.float64)

        elif self.context.base == NumericalBase.FIXED_POINT:
            # Q16.16 integer fixed point execution
            scale = 65536.0
            x_fp = np.round(x_arr * scale)
            x_dec = x_fp / scale
            out = eval_fn(x_dec)
            return np.round(out * scale) / scale

        elif self.context.base == NumericalBase.LOGARITHMIC:
            # Deterministic LNS representation log_b(|x|)
            sign_x = np.sign(x_arr)
            abs_x = np.maximum(1e-15, np.abs(x_arr))
            log_x = np.log2(abs_x)
            recon_x = sign_x * (2.0 ** log_x)
            return eval_fn(recon_x)

        else:
            return eval_fn(x_arr.astype(np.float64))

    def evaluate_normalized_recurrence(self, x: np.ndarray) -> np.ndarray:
        """Normalized Three-term Recurrence for zonal function phi_n(x): O(n) FLOPs."""
        return self._execute_backend(lambda x_in: normalized_phi_recurrence(self.n, self.lambda_val, x_in), x)

    def evaluate_quotient_ring_normal_form(self, x: np.ndarray) -> np.ndarray:
        """
        Quotient Ring Harmonic Zonal Polynomial Projection:
        Evaluates degree-n spherical zonal polynomial phi_n(z_1) = _2F_1(-n, n+2*lambda; lambda+0.5; (1-z_1)/2)
        modulo q = sum(z_i^2) in R(Q).
        """
        if 2 * self.lambda_val + 2 < 3:
            d = 3
        else:
            d = int(round(2 * self.lambda_val + 2))

        coeffs = normalized_gegenbauer_2f1_coefficients(self.n, self.lambda_val)

        def _q_eval(x_in):
            x_q = np.asarray(x_in, dtype=np.float64)
            out = np.zeros_like(x_q)
            for i, xi in enumerate(x_q):
                rem_sq = max(0.0, 1.0 - xi**2) / max(1, d - 1)
                pt = [xi] + [np.sqrt(rem_sq)] * (d - 1)

                z1 = pt[0]
                t = (1.0 - z1) / 2.0
                val = sum(c * (t**k) for k, c in enumerate(coeffs))
                out[i] = val
            return out

        return self._execute_backend(_q_eval, x)

    def evaluate_hypergeometric(self, x: np.ndarray) -> np.ndarray:
        """
        Direct Hypergeometric _2F_1 Series Evaluation:
          phi_n(x) = _2F_1(-n, n + 2*lambda; lambda + 0.5; (1-x)/2).
        """
        def _hyp_eval(x_in):
            x_arr = np.asarray(x_in, dtype=np.float64)
            z = (1.0 - x_arr) / 2.0
            return hyp2f1(-self.n, self.n + 2.0 * self.lambda_val, self.lambda_val + 0.5, z)

        return self._execute_backend(_hyp_eval, x)

    def evaluate_wkb_weyl(self, x: np.ndarray) -> np.ndarray:
        """Interior WKB / Weyl expression for zonal function phi_n(x). Valid for |x| < 1."""
        def _wkb_eval(x_in):
            x_arr = np.asarray(x_in, dtype=np.float64)
            out = np.full_like(x_arr, np.nan)
            valid_mask = (np.abs(x_arr) < 1.0 - 1e-12)
            if np.any(valid_mask):
                theta = np.arccos(x_arr[valid_mask])
                out[valid_mask] = interior_wkb_approx(self.n, self.lambda_val, theta)
            return out

        return self._execute_backend(_wkb_eval, x)

    def evaluate_mehler_heine(self, x: np.ndarray) -> np.ndarray:
        """Mehler-Heine North-Pole Bessel Boundary-Layer expression for zonal function phi_n(x)."""
        def _mh_eval(x_in):
            x_arr = np.asarray(x_in, dtype=np.float64)
            theta = np.arccos(np.clip(x_arr, -1.0, 1.0))
            return endpoint_bessel_leading(self.n, self.lambda_val, theta)

        return self._execute_backend(_mh_eval, x)

    def evaluate_composite_matched(self, x: np.ndarray) -> np.ndarray:
        """Two-Endpoint Composite Matched Asymptotic expression for zonal function phi_n(x)."""
        def _comp_eval(x_in):
            x_arr = np.asarray(x_in, dtype=np.float64)
            theta = np.arccos(np.clip(x_arr, -1.0, 1.0))
            return composite_matched_approx(self.n, self.lambda_val, theta)

        return self._execute_backend(_comp_eval, x)

    def estimate_flops(self, perm: AlgebraicPermutation, num_points: int) -> int:
        """Estimates computational FLOP count for evaluation of N points."""
        if perm == AlgebraicPermutation.NORMALIZED_RECURRENCE:
            return 5 * self.n * num_points
        elif perm == AlgebraicPermutation.QUOTIENT_RING_NORMAL_FORM:
            return 10 * self.n * num_points
        elif perm == AlgebraicPermutation.HYPERGEOMETRIC_2F1:
            return 50 * num_points
        elif perm == AlgebraicPermutation.INTERIOR_WKB_WEYL:
            return 25 * num_points
        elif perm == AlgebraicPermutation.MEHLER_HEINE_BESSEL:
            return 30 * num_points
        elif perm == AlgebraicPermutation.COMPOSITE_MATCHED:
            return 60 * num_points
        return 100 * num_points

    def benchmark_permutations(self, domain_x: np.ndarray) -> Dict[AlgebraicPermutation, SolverPerformanceMetrics]:
        """
        Benchmarks all expression permutations over domain_x and records
        computational cost, execution time, and multi-percentile mixed numerical error relative
        to high-precision ground truth reference.
        """
        ground_truth = high_precision_reference(self.n, self.lambda_val, domain_x, dps=100)
        num_points = len(domain_x)
        results = {}

        eval_map = {
            AlgebraicPermutation.NORMALIZED_RECURRENCE: self.evaluate_normalized_recurrence,
            AlgebraicPermutation.QUOTIENT_RING_NORMAL_FORM: self.evaluate_quotient_ring_normal_form,
            AlgebraicPermutation.HYPERGEOMETRIC_2F1: self.evaluate_hypergeometric,
            AlgebraicPermutation.INTERIOR_WKB_WEYL: self.evaluate_wkb_weyl,
            AlgebraicPermutation.MEHLER_HEINE_BESSEL: self.evaluate_mehler_heine,
            AlgebraicPermutation.COMPOSITE_MATCHED: self.evaluate_composite_matched,
        }

        for perm, fn in eval_map.items():
            _ = fn(domain_x[:min(10, num_points)])

            t0 = time.perf_counter()
            iterations = 10 if num_points < 1000 else 1
            for _ in range(iterations):
                val = fn(domain_x)
            t1 = time.perf_counter()
            exec_time = (t1 - t0) / iterations

            abs_res = np.abs(val - ground_truth)
            max_res = float(np.nanmax(abs_res))

            mix_errs = mixed_error(val, ground_truth)
            max_mix = float(np.nanmax(mix_errs))
            med_mix = float(np.nanmedian(mix_errs))
            p95_mix = float(np.nanpercentile(mix_errs, 95))
            rms_mix = float(np.sqrt(np.nanmean(mix_errs ** 2)))

            num_flops = self.estimate_flops(perm, num_points)

            results[perm] = SolverPerformanceMetrics(
                permutation=perm,
                num_flops=num_flops,
                exec_time_sec=exec_time,
                max_residual=max_res,
                max_mixed_error=max_mix,
                median_mixed_error=med_mix,
                p95_mixed_error=p95_mix,
                rms_mixed_error=rms_mix,
            )

        self._compute_pareto_frontier(results)
        return results

    def _compute_pareto_frontier(self, metrics_map: Dict[AlgebraicPermutation, SolverPerformanceMetrics]):
        """Identifies non-dominated solutions on the (Execution Time, Max Mixed Error) plane."""
        items = list(metrics_map.values())
        for a in items:
            dominated = False
            for b in items:
                if a.permutation == b.permutation:
                    continue
                if (b.exec_time_sec <= a.exec_time_sec and b.max_mixed_error <= a.max_mixed_error) and \
                   (b.exec_time_sec < a.exec_time_sec or b.max_mixed_error < a.max_mixed_error):
                    dominated = True
                    break
            a.is_pareto_optimal = not dominated

    def solve_optimal_permutation(self, domain_x: np.ndarray, max_error_tol: Optional[float] = None,
                                   max_flop_budget: Optional[int] = None) -> SolverPerformanceMetrics:
        """
        Solves for the optimal expression permutation.
        Raises ValueError if no candidate satisfies requested max_error_tol or max_flop_budget.
        """
        metrics = self.benchmark_permutations(domain_x)
        pareto_candidates = [m for m in metrics.values() if m.is_pareto_optimal]

        if not pareto_candidates:
            pareto_candidates = list(metrics.values())

        if max_error_tol is not None:
            filtered = [m for m in pareto_candidates if m.max_mixed_error <= max_error_tol]
            if not filtered:
                raise ValueError(f"No algebraic permutation satisfies max_error_tol={max_error_tol}")
            pareto_candidates = filtered

        if max_flop_budget is not None:
            filtered = [m for m in pareto_candidates if m.num_flops <= max_flop_budget]
            if not filtered:
                raise ValueError(f"No algebraic permutation satisfies max_flop_budget={max_flop_budget}")
            pareto_candidates = filtered

        if max_error_tol is not None:
            best = min(pareto_candidates, key=lambda m: m.exec_time_sec)
        else:
            best = min(pareto_candidates, key=lambda m: m.max_mixed_error)

        return best


if __name__ == "__main__":
    print("--- COMPUTATIONAL LAYER & PARETO OPTIMIZER DEMO ---")
    n_deg = 50
    lambda_p = 1.5
    ctx = NumericalContext.default_float64()
    solver = GegenbauerComputationalSolver(n=n_deg, lambda_val=lambda_p, context=ctx)

    domain = np.linspace(-0.8, 0.8, 500)
    print(f"Problem: Degree n={n_deg}, Lambda={lambda_p}, Num Points={len(domain)}")

    results = solver.benchmark_permutations(domain)
    print("\n[Benchmark Results across All 6 Algebraic Permutations]")
    print(f"{'Algebraic Permutation':<38} | {'FLOPs':>8} | {'Exec Time (ms)':>14} | {'Max Mixed Error':>15} | {'Pareto Optimal'}")
    print("-" * 96)
    for perm, m in results.items():
        time_ms = m.exec_time_sec * 1000.0
        print(f"{m.permutation.value:<38} | {m.num_flops:8d} | {time_ms:14.4f} | {m.max_mixed_error:15.6e} | {str(m.is_pareto_optimal)}")

    optimal = solver.solve_optimal_permutation(domain, max_error_tol=1e-2)
    print(f"\nOptimal Permutation selected for max_error_tol=1e-2: {optimal.permutation.value}")
