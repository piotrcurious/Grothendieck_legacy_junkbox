"""
Comprehensive Unit Test Suite for Gegenbauer Demystification Framework
========================================================================
Tests:
1. Formal Prolog Proof Knowledgebase Execution (via swipl)
2. Python Semiclassical Asymptotics & Error Convergence Rates
3. Computational Layer & Pareto Optimization Solver across bases/types
4. Quadric Hypersurface Algebraic Geometry & Normalized Jacobi Recurrence
5. High-Degree Stability & L2 Orthogonality Quadrature Integrals
"""

import subprocess
import math
import numpy as np
import pytest

from Gregenbauer_demistify.gegenbauer_asymptotics import (
    exact_gegenbauer,
    interior_wkb_approx,
    mehler_heine_bessel_approx,
    c_n_1_val,
    log_c_n_1,
    orthogonality_norm,
    verify_orthogonality_integral,
)
from Gregenbauer_demistify.computational_layer import (
    GegenbauerComputationalSolver,
    NumericalContext,
    AlgebraicPermutation,
    PrecisionType,
    NumericalBase,
    HAS_MPMATH,
)
from Gregenbauer_demistify.algebraic_geometry_combinatorics import (
    quadric_hilbert_polynomial,
    pieri_coefficients,
    normalized_jacobi_coefficients,
    pochhammer,
    schubert_intersection_coefficients,
)


def test_prolog_formal_proof():
    """Executes SWI-Prolog proof knowledgebase and verifies logical theorems pass."""
    cmd = ["swipl", "-g", "run_all_proofs", "-t", "halt", "Gregenbauer_demistify/gegenbauer_proof.pl"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"SWI-Prolog returned non-zero exit code: {res.stderr}"
    assert "PROOF COMPLETED SUCCESSFULLY WITH ALL ASSERTIONS VERIFIED EXACTLY!" in res.stdout
    assert "Symmetric Space: so(5)/so(4)" in res.stdout
    assert "Three-Term Recurrence & Numerical Evaluation Verification" in res.stdout
    assert "Representation Geometry of Null Quadric Q^{d-2} c P^{d-1}" in res.stdout
    assert "Antipodal Parity Symmetry" in res.stdout
    assert "Demystification of the Singular Scaling Limit (4-Fold Unification)" in res.stdout
    assert "Matched Asymptotic Overlap Verification (Regime III" in res.stdout


def test_high_degree_stability():
    """Verifies high degree n=1000, 2000 evaluation stability in log-space."""
    for n in [1000, 2000]:
        log_val = log_c_n_1(n, 1.5)
        assert np.isfinite(log_val)
        val = c_n_1_val(n, 1.5)
        assert np.isfinite(val)
        assert val > 0.0


def test_orthogonality_norm_and_quadrature_integral():
    """Verifies L2 orthogonality integral and norm formula h_n."""
    lambda_val = 1.5
    h_2_exact = orthogonality_norm(2, lambda_val)
    h_2_num = verify_orthogonality_integral(2, 2, lambda_val)
    assert abs(h_2_exact - h_2_num) / h_2_exact < 1e-4

    h_23_num = verify_orthogonality_integral(2, 3, lambda_val)
    assert abs(h_23_num) < 1e-10


def test_quadric_hilbert_polynomial_and_normalization():
    """Verifies C_n^(lambda)(1) = (lambda / (n + lambda)) * dim H^0(Q_{d-2}, O(n))."""
    d = 5
    n = 6
    lambda_val = (d - 2) / 2.0

    h0 = quadric_hilbert_polynomial(d, n)
    c_n_1 = c_n_1_val(n, lambda_val)

    expected_h0 = math.comb(n + d - 1, d - 1) - math.comb(n + d - 3, d - 1)
    assert h0 == expected_h0

    rel_c_n_1 = (lambda_val / (n + lambda_val)) * h0
    assert abs(c_n_1 - rel_c_n_1) < 1e-12


def test_normalized_jacobi_coefficients_sum_identity():
    """Verifies that normalized Jacobi coefficients a_n + b_n = 1 for any n, lambda."""
    for n in range(1, 10):
        for lambda_val in [0.5, 1.0, 1.5, 2.5]:
            a_n, b_n = normalized_jacobi_coefficients(n, lambda_val)
            assert abs((a_n + b_n) - 1.0) < 1e-12


def test_pochhammer_and_hypergeometric_coefficients():
    """Tests rising Pochhammer symbol and hypergeometric series expansion."""
    assert pochhammer(3.0, 4) == 360.0

    n = 2
    lambda_val = 1.5
    coeffs = schubert_intersection_coefficients(n, lambda_val)
    assert len(coeffs) == 3
    assert abs(coeffs[0] - 1.0) < 1e-12
    assert abs(coeffs[1] - (-5.0)) < 1e-12
    assert abs(coeffs[2] - 5.0) < 1e-12

    # Verify polynomial value at x = 0
    c_n_1 = c_n_1_val(n, lambda_val)
    t = 0.5  # (1-0)/2
    poly_val = c_n_1 * sum(c * (t**k) for k, c in enumerate(coeffs))
    exact_val = exact_gegenbauer(n, lambda_val, np.array([0.0]))[0]
    assert abs(poly_val - exact_val) < 1e-12


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

    assert errors[0] < 0.02
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

    assert errors[0] < 1e-3
    ratio = errors[2] / errors[0]
    assert ratio < 0.1


def test_computational_solver_benchmark_and_pareto():
    """Tests computational layer benchmarking and Pareto optimization solver."""
    n_deg = 100
    lambda_p = 1.5
    ctx = NumericalContext.default_float64()
    solver = GegenbauerComputationalSolver(n=n_deg, lambda_val=lambda_p, context=ctx)

    domain = np.linspace(0.2, 0.98, 200)
    results = solver.benchmark_permutations(domain)

    assert len(results) == 6
    assert AlgebraicPermutation.CLENSHAW_RECURRENCE in results
    assert AlgebraicPermutation.HYPERGEOMETRIC_2F1 in results

    pareto_candidates = [m for m in results.values() if m.is_pareto_optimal]
    assert len(pareto_candidates) >= 1

    opt_acc = solver.solve_optimal_permutation(domain, max_error_tol=1e-5)
    assert opt_acc.max_relative_error <= 1e-5

    opt_budget = solver.solve_optimal_permutation(domain, max_flop_budget=15000)
    assert opt_budget.num_flops <= 15000


def test_computational_solver_numerical_bases():
    """Tests computational solver across Fixed-Point Q16.16 and Logarithmic LNS contexts."""
    n_deg = 20
    lambda_p = 1.5
    domain = np.linspace(0.1, 0.9, 50)

    ctx_fp = NumericalContext.fixed_point_q16()
    solver_fp = GegenbauerComputationalSolver(n=n_deg, lambda_val=lambda_p, context=ctx_fp)
    res_fp = solver_fp.evaluate_clenshaw_recurrence(domain)
    assert ctx_fp.base == NumericalBase.FIXED_POINT
    assert len(res_fp) == 50

    ctx_lns = NumericalContext.logarithmic_lns()
    solver_lns = GegenbauerComputationalSolver(n=n_deg, lambda_val=lambda_p, context=ctx_lns)
    res_lns = solver_lns.evaluate_clenshaw_recurrence(domain)
    assert ctx_lns.base == NumericalBase.LOGARITHMIC
    assert len(res_lns) == 50


def test_mpmath_arbitrary_precision():
    """Tests mpmath arbitrary precision evaluation if mpmath is available."""
    if not HAS_MPMATH:
        pytest.skip("mpmath not installed")

    n_deg = 10
    lambda_p = 1.5
    ctx_mp = NumericalContext.mpmath_arbitrary(dps=30)
    solver = GegenbauerComputationalSolver(n=n_deg, lambda_val=lambda_p, context=ctx_mp)
    domain = np.array([0.2, 0.5, 0.8])
    res = solver.evaluate_mpmath_arbitrary(domain)
    exact = exact_gegenbauer(n_deg, lambda_p, domain)
    np.testing.assert_allclose(res, exact, rtol=1e-10)
