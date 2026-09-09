"""
Comprehensive Unit Test Suite for Gegenbauer Demystification Framework
========================================================================
Tests:
1. Formal Prolog Proof Knowledgebase Execution (via swipl)
2. Python Semiclassical Asymptotics & Error Convergence Rates
3. Computational Layer & Pareto Optimization Solver across bases/types
"""

import subprocess
import numpy as np
import pytest

from Gregenbauer_demistify.gegenbauer_asymptotics import (
    exact_gegenbauer,
    interior_wkb_approx,
    mehler_heine_bessel_approx,
    c_n_1_val,
)
from Gregenbauer_demistify.computational_layer import (
    GegenbauerComputationalSolver,
    NumericalContext,
    AlgebraicPermutation,
    PrecisionType,
    NumericalBase,
)


def test_prolog_formal_proof():
    """Executes SWI-Prolog proof knowledgebase and verifies logical theorems pass."""
    cmd = ["swipl", "-g", "run_all_proofs", "-t", "halt", "Gregenbauer_demistify/gegenbauer_proof.pl"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"SWI-Prolog returned non-zero exit code: {res.stderr}"
    assert "PROOF COMPLETED SUCCESSFULLY WITH ALL THEOREMS VERIFIED LOGICALLY!" in res.stdout
    assert "Symmetric Space: so(5)/so(4)" in res.stdout


def test_wkb_interior_asymptotic_convergence():
    """Verifies that interior WKB relative error decreases as O(1/n)."""
    lambda_val = 1.5
    theta_val = np.pi / 4.0
    x_val = np.cos(theta_val)

    errors = []
    n_values = [50, 100, 200]

    for n in n_values:
        exact = exact_gegenbauer(n, lambda_val, np.array([x_val]))[0]
        wkb = interior_wkb_approx(n, lambda_val, np.array([theta_val]))[0]
        rel_err = abs(exact - wkb) / abs(exact)
        errors.append(rel_err)

    # Relative error should be under 2% for n >= 50
    assert errors[0] < 0.02
    # Relative error should decrease as n increases
    assert errors[-1] < errors[0]


def test_mehler_heine_bessel_convergence():
    """Verifies that endpoint Bessel kernel error decreases as O(1/n^2)."""
    lambda_val = 1.5
    z_fix = 2.5
    n_values = [50, 100, 200]
    errors = []

    for n in n_values:
        K = n + lambda_val
        theta_end = z_fix / K
        x_end = np.cos(theta_end)

        c_n_1 = c_n_1_val(n, lambda_val)
        exact_ratio = exact_gegenbauer(n, lambda_val, np.array([x_end]))[0] / c_n_1
        bessel_ratio = mehler_heine_bessel_approx(n, lambda_val, np.array([theta_end]))[0] / c_n_1

        abs_err = abs(exact_ratio - bessel_ratio)
        errors.append(abs_err)

    # Error at z=2.5 should be < 1e-3 for n >= 50
    assert errors[0] < 1e-3
    # Quadratic decrease: error at n=200 should be ~1/16 of error at n=50
    ratio = errors[2] / errors[0]
    assert ratio < 0.1  # Approx 1/16 = 0.0625


def test_computational_solver_benchmark_and_pareto():
    """Tests computational layer benchmarking and Pareto optimization solver."""
    n_deg = 100
    lambda_p = 1.5
    ctx = NumericalContext.default_float64()
    solver = GegenbauerComputationalSolver(n=n_deg, lambda_val=lambda_p, context=ctx)

    domain = np.linspace(0.2, 0.98, 200)
    results = solver.benchmark_permutations(domain)

    # Verify all expected permutations were evaluated
    assert AlgebraicPermutation.CLENSHAW_RECURRENCE in results
    assert AlgebraicPermutation.HYPERGEOMETRIC_2F1 in results
    assert AlgebraicPermutation.INTERIOR_WKB_WEYL in results
    assert AlgebraicPermutation.MEHLER_HEINE_BESSEL in results
    assert AlgebraicPermutation.COMPOSITE_MATCHED in results

    # Verify Pareto frontier identifies optimal candidates
    pareto_candidates = [m for m in results.values() if m.is_pareto_optimal]
    assert len(pareto_candidates) >= 1

    # Test solver selection under accuracy constraint
    opt_acc = solver.solve_optimal_permutation(domain, max_error_tol=1e-5)
    assert opt_acc.max_relative_error <= 1e-5

    # Test solver selection under budget constraint
    opt_budget = solver.solve_optimal_permutation(domain, max_flop_budget=15000)
    assert opt_budget.num_flops <= 15000


def test_computational_solver_float32_context():
    """Tests computational solver with float32 precision context."""
    n_deg = 50
    lambda_p = 2.0
    ctx = NumericalContext.float32()
    solver = GegenbauerComputationalSolver(n=n_deg, lambda_val=lambda_p, context=ctx)

    domain = np.linspace(0.1, 0.9, 100)
    results = solver.benchmark_permutations(domain)

    assert solver.context.precision == PrecisionType.FLOAT32
    assert solver.context.eps == 1.19e-7
    assert len(results) == 5
