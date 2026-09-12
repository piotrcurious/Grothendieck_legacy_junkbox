"""
Computational Layer and Pareto Optimization Solver for Gegenbauer Polynomials
================================================================================
This module provides a computational framework taking numerical bases and types into
account, and selects optimal algebraic geometry expression permutations on the
2D Optimization (Computational Cost) vs Numerical Error plane.

Features:
- Representation of Numerical Types (float32, float64, float128, mpmath)
  and Numerical Bases (Base 2, Base 10, Fixed-Point, Logarithmic).
- Robust Mixed Error Metric E = |f_approx - f_ref| / (atol + rtol * |f_ref|)
- Algebraic Geometry Expression Permutations:
    1. Normalized Three-Term Recurrence phi_n(x)
    2. Quotient Ring Normal Form Polynomial Remainder
    3. Hypergeometric _2F_1 Series Expansion
    4. Interior WKB / Weyl Semiclassical Expression
    5. Mehler-Heine Bessel Boundary-Layer Expression
    6. Composite Matched Asymptotic Expression
- Pareto Solver finding optimal expression permutations under user speed/accuracy constraints.
"""

from dataclasses import dataclass
from enum import Enum
import time
from typing import Dict, List, Optional

import numpy as np
from scipy.special import eval_gegenbauer, gamma, gammaln, jv, hyp2f1

try:
    import mpmath
    HAS_MPMATH = True
except ImportError:
    HAS_MPMATH = False

try:
    from algebraic_geometry_combinatorics import QuadricQuotientPolynomial, normalized_jacobi_coefficients
    from gegenbauer_asymptotics import normalized_phi_recurrence, mehler_heine_bessel_approx, interior_wkb_approx, composite_matched_approx, c_n_1_val
except ModuleNotFoundError:
    from Gregenbauer_demistify.algebraic_geometry_combinatorics import QuadricQuotientPolynomial, normalized_jacobi_coefficients
    from Gregenbauer_demistify.gegenbauer_asymptotics import normalized_phi_recurrence, mehler_heine_bessel_approx, interior_wkb_approx, composite_matched_approx, c_n_1_val


def mixed_error(approx: np.ndarray, ref: np.ndarray, atol: float = 1e-14, rtol: float = 1e-10) -> np.ndarray:
    """Computes robust mixed error to handle near-zero values near polynomial roots."""
    return np.abs(approx - ref) / (atol + rtol * np.abs(ref))


class NumericalBase(Enum):
    BASE_2 = "Base 2 (IEEE Binary)"
    BASE_10 = "Base 10 (Decimal)"
    FIXED_POINT = "Fixed-Point (Q16.16)"
    LOGARITHMIC = "Logarithmic Number System (LNS)"


class PrecisionType(Enum):
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    FLOAT128 = "float128"
    ARBITRARY = "mpmath_arbitrary"


@dataclass
class NumericalContext:
    base: NumericalBase = NumericalBase.BASE_2
    precision: PrecisionType = PrecisionType.FLOAT64
    bits: int = 64
    eps: float = 2.22e-16

    @classmethod
    def default_float64(cls):
        return cls(base=NumericalBase.BASE_2, precision=PrecisionType.FLOAT64, bits=64, eps=2.22e-16)

    @classmethod
    def float32(cls):
        return cls(base=NumericalBase.BASE_2, precision=PrecisionType.FLOAT32, bits=32, eps=1.19e-7)

    @classmethod
    def fixed_point_q16(cls):
        return cls(base=NumericalBase.FIXED_POINT, precision=PrecisionType.FLOAT32, bits=32, eps=1.52e-5)

    @classmethod
    def logarithmic_lns(cls):
        return cls(base=NumericalBase.LOGARITHMIC, precision=PrecisionType.FLOAT32, bits=32, eps=1e-4)

    @classmethod
    def mpmath_arbitrary(cls, dps: int = 50):
        return cls(base=NumericalBase.BASE_10, precision=PrecisionType.ARBITRARY, bits=dps * 4, eps=10**(-dps))


class AlgebraicPermutation(Enum):
    NORMALIZED_RECURRENCE = "Normalized Recurrence phi_n(x)"
    QUOTIENT_RING_NORMAL_FORM = "Quotient Ring Normal Form Remainder"
    HYPERGEOMETRIC_2F1 = "Hypergeometric _2F1 Series"
    INTERIOR_WKB_WEYL = "Interior WKB / Weyl Semiclassical"
    MEHLER_HEINE_BESSEL = "Mehler-Heine Bessel Boundary-Layer"
    COMPOSITE_MATCHED = "Composite Matched Asymptotic"


@dataclass
class SolverPerformanceMetrics:
    permutation: AlgebraicPermutation
    num_flops: int
    exec_time_sec: float
    estimated_error: float
    max_residual: float
    max_mixed_error: float
    is_pareto_optimal: bool = False


class GegenbauerComputationalSolver:
    """
    Evaluates equivalent algebraic geometry permutations for Gegenbauer polynomials
    and zonal functions under specific numerical contexts and solves for Pareto-optimal expressions.
    """

    def __init__(self, n: int, lambda_val: float, context: NumericalContext = None):
        self.n = n
        self.lambda_val = lambda_val
        self.context = context or NumericalContext.default_float64()

    def _apply_context_quantization(self, arr: np.ndarray) -> np.ndarray:
        """Simulates precision and numerical base constraints."""
        if self.context.base == NumericalBase.FIXED_POINT:
            scale = 65536.0
            return np.round(arr * scale) / scale
        elif self.context.base == NumericalBase.LOGARITHMIC:
            noise = 1.0 + np.random.normal(0, self.context.eps, size=arr.shape)
            return arr * noise
        elif self.context.precision == PrecisionType.FLOAT32:
            return arr.astype(np.float32).astype(np.float64)
        return arr

    def evaluate_normalized_recurrence(self, x: np.ndarray) -> np.ndarray:
        """Normalized Three-term Recurrence for phi_n(x): O(n) FLOPs."""
        x_q = self._apply_context_quantization(np.asarray(x, dtype=np.float64))
        phi_vals = normalized_phi_recurrence(self.n, self.lambda_val, x_q)
        return self._apply_context_quantization(phi_vals)

    def evaluate_quotient_ring_normal_form(self, x: np.ndarray) -> np.ndarray:
        """Quotient Ring Polynomial Remainder normal form evaluation."""
        x_q = self._apply_context_quantization(np.asarray(x, dtype=np.float64))
        d = int(2 * self.lambda_val + 2)

        # Build z_1^n modulo q in R(Q)
        poly = QuadricQuotientPolynomial(d, {(0,) * d: 1.0})
        for _ in range(self.n):
            poly = poly.multiply_by_x(0)

        # Evaluate on zonal points (x_q, sqrt(1-x_q^2)/(d-1), ...)
        out = np.zeros_like(x_q)
        for i, xi in enumerate(x_q):
            rem_sq = max(0.0, 1.0 - xi**2) / max(1, d - 1)
            pt = [xi] + [np.sqrt(rem_sq)] * (d - 1)
            out[i] = poly.evaluate(pt)

        # Normalize to zonal function phi_n(x)
        c_n_1 = c_n_1_val(self.n, self.lambda_val) if self.n > 0 else 1.0
        return self._apply_context_quantization(out)

    def evaluate_hypergeometric(self, x: np.ndarray) -> np.ndarray:
        """Hypergeometric _2F1 Series Evaluation."""
        x_q = self._apply_context_quantization(np.asarray(x, dtype=np.float64))
        z = (1.0 - x_q) / 2.0
        c_n_1 = c_n_1_val(self.n, self.lambda_val)
        h_val = hyp2f1(-self.n, self.n + 2.0 * self.lambda_val, self.lambda_val + 0.5, z)
        return self._apply_context_quantization((c_n_1 * h_val) / c_n_1)

    def evaluate_wkb_weyl(self, x: np.ndarray) -> np.ndarray:
        """Interior WKB / Weyl expression for zonal function phi_n(x)."""
        x_q = self._apply_context_quantization(np.clip(x, -0.999999, 0.999999))
        theta = np.arccos(x_q)
        c_n_1 = c_n_1_val(self.n, self.lambda_val)
        wkb_c = interior_wkb_approx(self.n, self.lambda_val, theta)
        return self._apply_context_quantization(wkb_c / c_n_1)

    def evaluate_mehler_heine(self, x: np.ndarray) -> np.ndarray:
        """Mehler-Heine Bessel Boundary-Layer expression for zonal function phi_n(x)."""
        x_q = self._apply_context_quantization(np.clip(x, -1.0, 1.0))
        theta = np.arccos(x_q)
        c_n_1 = c_n_1_val(self.n, self.lambda_val)
        bessel_c = mehler_heine_bessel_approx(self.n, self.lambda_val, theta)
        return self._apply_context_quantization(bessel_c / c_n_1)

    def evaluate_composite_matched(self, x: np.ndarray) -> np.ndarray:
        """Composite Matched Asymptotic expression for zonal function phi_n(x)."""
        x_q = self._apply_context_quantization(np.clip(x, -0.999999, 0.999999))
        theta = np.arccos(x_q)
        c_n_1 = c_n_1_val(self.n, self.lambda_val)
        comp_c = composite_matched_approx(self.n, self.lambda_val, theta)
        return self._apply_context_quantization(comp_c / c_n_1)

    def estimate_flops(self, perm: AlgebraicPermutation, num_points: int) -> int:
        """Estimates computational FLOP count for evaluation of N points."""
        if perm == AlgebraicPermutation.NORMALIZED_RECURRENCE:
            return 5 * self.n * num_points
        elif perm == AlgebraicPermutation.QUOTIENT_RING_NORMAL_FORM:
            return 10 * self.n * num_points
        elif perm == AlgebraicPermutation.HYPERGEOMETRIC_2F1:
            return 20 * num_points
        elif perm == AlgebraicPermutation.INTERIOR_WKB_WEYL:
            return 12 * num_points
        elif perm == AlgebraicPermutation.MEHLER_HEINE_BESSEL:
            return 25 * num_points
        elif perm == AlgebraicPermutation.COMPOSITE_MATCHED:
            return 40 * num_points
        return 100 * num_points

    def benchmark_permutations(self, domain_x: np.ndarray) -> Dict[AlgebraicPermutation, SolverPerformanceMetrics]:
        """
        Benchmarks all expression permutations over domain_x and records
        computational cost, execution time, and residual numerical error relative
        to double-precision ground truth zonal function phi_n(x).
        """
        c_n_1 = c_n_1_val(self.n, self.lambda_val)
        ground_truth = eval_gegenbauer(self.n, self.lambda_val, domain_x) / c_n_1
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
            t0 = time.perf_counter()
            iterations = 10 if num_points < 1000 else 1
            for _ in range(iterations):
                val = fn(domain_x)
            t1 = time.perf_counter()
            exec_time = (t1 - t0) / iterations

            abs_res = np.abs(val - ground_truth)
            max_res = float(np.max(abs_res))

            # Robust Mixed Error calculation
            mix_errs = mixed_error(val, ground_truth)
            max_mix_err = float(np.max(mix_errs))
            mean_mix_err = float(np.mean(mix_errs))

            num_flops = self.estimate_flops(perm, num_points)
            estimated_error = mean_mix_err + (num_flops * self.context.eps)

            results[perm] = SolverPerformanceMetrics(
                permutation=perm,
                num_flops=num_flops,
                exec_time_sec=exec_time,
                estimated_error=estimated_error,
                max_residual=max_res,
                max_mixed_error=max_mix_err,
            )

        self._compute_pareto_frontier(results)
        return results

    def _compute_pareto_frontier(self, metrics_map: Dict[AlgebraicPermutation, SolverPerformanceMetrics]):
        """Identifies non-dominated solutions on the (Cost, Mixed Error) plane."""
        items = list(metrics_map.values())
        for a in items:
            dominated = False
            for b in items:
                if a.permutation == b.permutation:
                    continue
                if (b.num_flops <= a.num_flops and b.max_mixed_error <= a.max_mixed_error) and \
                   (b.num_flops < a.num_flops or b.max_mixed_error < a.max_mixed_error):
                    dominated = True
                    break
            a.is_pareto_optimal = not dominated

    def solve_optimal_permutation(self, domain_x: np.ndarray, max_error_tol: Optional[float] = None,
                                   max_flop_budget: Optional[int] = None) -> SolverPerformanceMetrics:
        """
        Solves for the optimal algebraic geometry expression permutation based on
        user speed/accuracy constraints.
        """
        metrics = self.benchmark_permutations(domain_x)
        pareto_candidates = [m for m in metrics.values() if m.is_pareto_optimal]

        if not pareto_candidates:
            pareto_candidates = list(metrics.values())

        if max_error_tol is not None:
            filtered = [m for m in pareto_candidates if m.max_mixed_error <= max_error_tol]
            if filtered:
                pareto_candidates = filtered

        if max_flop_budget is not None:
            filtered = [m for m in pareto_candidates if m.num_flops <= max_flop_budget]
            if filtered:
                pareto_candidates = filtered

        if max_error_tol is not None:
            best = min(pareto_candidates, key=lambda m: m.num_flops)
        else:
            best = min(pareto_candidates, key=lambda m: m.max_mixed_error)

        return best


if __name__ == "__main__":
    print("--- COMPUTATIONAL LAYER & PARETO OPTIMIZER DEMO ---")
    n_deg = 50
    lambda_p = 1.5
    ctx = NumericalContext.default_float64()
    solver = GegenbauerComputationalSolver(n=n_deg, lambda_val=lambda_p, context=ctx)

    domain = np.linspace(0.5, 0.999, 500)
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
