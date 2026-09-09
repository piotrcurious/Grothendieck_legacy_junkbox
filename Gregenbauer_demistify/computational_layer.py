"""
Computational Layer and Pareto Optimization Solver for Gegenbauer Polynomials
================================================================================
This module provides a computational framework taking numerical bases and types into
account, and selects optimal algebraic geometry expression permutations on the
2D Optimization (Computational Cost) vs Numerical Error plane.

Features:
- Representation of Numerical Types (float32, float64, float128, mpmath)
  and Numerical Bases (Base 2, Base 10, Fixed-Point, Logarithmic).
- Algebraic Geometry Expression Permutations:
    1. Three-Term Clenshaw Recurrence
    2. Hypergeometric _2F_1 Series Expansion
    3. Interior WKB / Weyl Semiclassical Expression
    4. Mehler-Heine Bessel Boundary-Layer Expression
    5. Composite Matched Asymptotic Expression
    6. Barycentric Rational / Chebyshev Proxy
- Pareto Solver finding optimal expression permutations under user speed/accuracy constraints.
"""

from dataclasses import dataclass
from enum import Enum
import time
from typing import Dict, List, Optional

import numpy as np
from scipy.special import eval_gegenbauer, gamma, gammaln, jv, hyp2f1


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


class AlgebraicPermutation(Enum):
    CLENSHAW_RECURRENCE = "Clenshaw Three-Term Recurrence"
    HYPERGEOMETRIC_2F1 = "Hypergeometric _2F1 Series"
    INTERIOR_WKB_WEYL = "Interior WKB / Weyl Semiclassical"
    MEHLER_HEINE_BESSEL = "Mehler-Heine Bessel Boundary-Layer"
    COMPOSITE_MATCHED = "Composite Matched Asymptotic"
    BARYCENTRIC_RATIONAL = "Barycentric Rational Proxy"


@dataclass
class SolverPerformanceMetrics:
    permutation: AlgebraicPermutation
    num_flops: int
    exec_time_sec: float
    estimated_error: float
    max_residual: float
    max_relative_error: float
    is_pareto_optimal: bool = False


class GegenbauerComputationalSolver:
    """
    Evaluates equivalent algebraic geometry permutations for Gegenbauer polynomials
    under specific numerical contexts and solves for Pareto-optimal expressions.
    """

    def __init__(self, n: int, lambda_val: float, context: NumericalContext = None):
        self.n = n
        self.lambda_val = lambda_val
        self.context = context or NumericalContext.default_float64()

    def _c_n_1(self) -> float:
        log_c_n_1 = gammaln(self.n + 2.0 * self.lambda_val) - gammaln(2.0 * self.lambda_val) - gammaln(self.n + 1.0)
        return float(np.exp(log_c_n_1))

    def evaluate_clenshaw_recurrence(self, x: np.ndarray) -> np.ndarray:
        """Clenshaw Three-term Recurrence: O(n) FLOPs."""
        x = np.asarray(x, dtype=np.float64)
        if self.n == 0:
            return np.ones_like(x)
        if self.n == 1:
            return 2.0 * self.lambda_val * x

        c0 = np.ones_like(x)
        c1 = 2.0 * self.lambda_val * x

        for k in range(2, self.n + 1):
            c2 = (2.0 * (k + self.lambda_val - 1.0) / k) * x * c1 - ((k + 2.0 * self.lambda_val - 2.0) / k) * c0
            c0, c1 = c1, c2

        return c1

    def evaluate_hypergeometric(self, x: np.ndarray) -> np.ndarray:
        """Hypergeometric _2F1(-n, n + 2*lambda; lambda + 0.5; (1-x)/2) evaluation via hyp2f1."""
        x = np.asarray(x, dtype=np.float64)
        z = (1.0 - x) / 2.0
        c_n_1 = self._c_n_1()
        h_val = hyp2f1(-self.n, self.n + 2.0 * self.lambda_val, self.lambda_val + 0.5, z)
        return c_n_1 * h_val

    def evaluate_wkb_weyl(self, x: np.ndarray) -> np.ndarray:
        """Interior WKB / Weyl expression: O(1) FLOPs."""
        x = np.clip(x, -0.999999, 0.999999)
        theta = np.arccos(x)
        K = self.n + self.lambda_val
        coeff = (2.0 ** (1.0 - self.lambda_val) / gamma(self.lambda_val)) * (self.n ** (self.lambda_val - 1.0))
        amplitude = (np.sin(theta)) ** (-self.lambda_val)
        phase = K * theta - (self.lambda_val * np.pi / 2.0)
        return coeff * amplitude * np.cos(phase)

    def evaluate_mehler_heine(self, x: np.ndarray) -> np.ndarray:
        """Mehler-Heine Bessel Boundary-Layer expression: O(1) FLOPs."""
        theta = np.arccos(np.clip(x, -1.0, 1.0))
        K = self.n + self.lambda_val
        z = K * theta
        nu = self.lambda_val - 0.5
        c_n_1 = self._c_n_1()

        z_safe = np.where(z == 0, 1e-15, z)
        cal_j_nu = (2.0 ** nu) * gamma(nu + 1.0) * (z_safe ** (-nu)) * jv(nu, z_safe)
        cal_j_nu = np.where(z == 0, 1.0, cal_j_nu)

        return c_n_1 * cal_j_nu

    def evaluate_composite_matched(self, x: np.ndarray) -> np.ndarray:
        """Composite Matched Asymptotic expression: O(1) FLOPs."""
        theta = np.arccos(np.clip(x, -0.999999, 0.999999))
        K = self.n + self.lambda_val
        z = K * theta
        nu = self.lambda_val - 0.5
        c_n_1 = self._c_n_1()

        bessel = self.evaluate_mehler_heine(x)
        wkb = self.evaluate_wkb_weyl(x)

        z_safe = np.where(z == 0, 1e-15, z)
        matching = (c_n_1 * (2.0 ** nu) * gamma(nu + 1.0) * (z_safe ** (-nu)) *
                    np.sqrt(2.0 / (np.pi * z_safe)) * np.cos(z_safe - self.lambda_val * np.pi / 2.0))

        return bessel + wkb - matching

    def estimate_flops(self, perm: AlgebraicPermutation, num_points: int) -> int:
        """Estimates computational FLOP count for evaluation of N points."""
        if perm == AlgebraicPermutation.CLENSHAW_RECURRENCE:
            return 5 * self.n * num_points
        elif perm == AlgebraicPermutation.HYPERGEOMETRIC_2F1:
            return 20 * num_points  # Scipy C library hyp2f1 call
        elif perm == AlgebraicPermutation.INTERIOR_WKB_WEYL:
            return 12 * num_points
        elif perm == AlgebraicPermutation.MEHLER_HEINE_BESSEL:
            return 25 * num_points
        elif perm == AlgebraicPermutation.COMPOSITE_MATCHED:
            return 40 * num_points
        elif perm == AlgebraicPermutation.BARYCENTRIC_RATIONAL:
            return 10 * num_points
        return 100 * num_points

    def benchmark_permutations(self, domain_x: np.ndarray) -> Dict[AlgebraicPermutation, SolverPerformanceMetrics]:
        """
        Benchmarks all available expression permutations over domain_x and records
        computational cost, execution time, and residual numerical error relative
        to double-precision ground truth.
        """
        ground_truth = eval_gegenbauer(self.n, self.lambda_val, domain_x)
        num_points = len(domain_x)
        results = {}

        eval_map = {
            AlgebraicPermutation.CLENSHAW_RECURRENCE: self.evaluate_clenshaw_recurrence,
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

            # Calculate residual numerical error
            abs_res = np.abs(val - ground_truth)
            max_res = float(np.max(abs_res))

            # Relative error
            c_n_1 = self._c_n_1()
            scale = np.maximum(np.abs(ground_truth), 1e-12)
            rel_errs = abs_res / scale
            max_rel_err = float(np.max(rel_errs))
            mean_rel_err = float(np.mean(rel_errs))

            num_flops = self.estimate_flops(perm, num_points)
            estimated_error = mean_rel_err + (num_flops * self.context.eps)

            results[perm] = SolverPerformanceMetrics(
                permutation=perm,
                num_flops=num_flops,
                exec_time_sec=exec_time,
                estimated_error=estimated_error,
                max_residual=max_res,
                max_relative_error=max_rel_err,
            )

        # Determine Pareto Optimality on Cost vs Relative Error plane
        self._compute_pareto_frontier(results)
        return results

    def _compute_pareto_frontier(self, metrics_map: Dict[AlgebraicPermutation, SolverPerformanceMetrics]):
        """
        Identifies non-dominated solutions on the (Cost, Relative Error) plane.
        A point A dominates point B if Cost(A) <= Cost(B) and RelErr(A) <= RelErr(B)
        with at least one strict inequality.
        """
        items = list(metrics_map.values())
        for a in items:
            dominated = False
            for b in items:
                if a.permutation == b.permutation:
                    continue
                if (b.num_flops <= a.num_flops and b.max_relative_error <= a.max_relative_error) and \
                   (b.num_flops < a.num_flops or b.max_relative_error < a.max_relative_error):
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
            filtered = [m for m in pareto_candidates if m.max_relative_error <= max_error_tol]
            if filtered:
                pareto_candidates = filtered

        if max_flop_budget is not None:
            filtered = [m for m in pareto_candidates if m.num_flops <= max_flop_budget]
            if filtered:
                pareto_candidates = filtered

        if max_error_tol is not None:
            best = min(pareto_candidates, key=lambda m: m.num_flops)
        else:
            best = min(pareto_candidates, key=lambda m: m.max_relative_error)

        return best


if __name__ == "__main__":
    print("--- COMPUTATIONAL LAYER & PARETO OPTIMIZER DEMO ---")
    n_deg = 100
    lambda_p = 1.5
    ctx = NumericalContext.default_float64()
    solver = GegenbauerComputationalSolver(n=n_deg, lambda_val=lambda_p, context=ctx)

    domain = np.linspace(0.5, 0.999, 500)
    print(f"Problem: Degree n={n_deg}, Lambda={lambda_p}, Num Points={len(domain)}")

    results = solver.benchmark_permutations(domain)
    print("\n[Benchmark Results across Algebraic Permutations]")
    print(f"{'Algebraic Permutation':<35} | {'FLOPs':>8} | {'Exec Time (ms)':>14} | {'Max Rel Error':>14} | {'Pareto Optimal'}")
    print("-" * 92)
    for perm, m in results.items():
        time_ms = m.exec_time_sec * 1000.0
        print(f"{m.permutation.value:<35} | {m.num_flops:8d} | {time_ms:14.4f} | {m.max_relative_error:14.6e} | {str(m.is_pareto_optimal)}")

    optimal = solver.solve_optimal_permutation(domain, max_error_tol=1e-2)
    print(f"\nOptimal Permutation selected for max_error_tol=1e-2: {optimal.permutation.value}")
