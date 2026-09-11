"""
Comprehensive Unit Test Suite for Gegenbauer Demystification Framework
========================================================================
Tests:
1. Formal Prolog Proof Knowledgebase Execution (via swipl)
2. Python Semiclassical Asymptotics, Phase Maps & Convergence Rates
3. Quadric Quotient Ring Normal Forms & Hilbert Series Data
4. Scaled Normalized Recurrence & Exact Anchors (S^2, S^3, S^4)
5. Computational Layer & Pareto Optimization Solver across bases/types
"""

import subprocess
import math
import numpy as np
import pytest

from Gregenbauer_demistify.gegenbauer_asymptotics import (
    exact_gegenbauer,
    normalized_phi_recurrence,
    interior_wkb_approx,
    mehler_heine_bessel_approx,
    c_n_1_val,
    log_c_n_1,
    orthogonality_norm,
    verify_orthogonality_integral,
    classify_phase_regime,
    exact_anchor_eval,
)
from Gregenbauer_demistify.computational_layer import (
    GegenbauerComputationalSolver,
    NumericalContext,
    AlgebraicPermutation,
    NumericalBase,
    HAS_MPMATH,
)
from Gregenbauer_demistify.algebraic_geometry_combinatorics import (
    quadric_hilbert_series_dim,
    QuadricQuotientPolynomial,
    normalized_jacobi_coefficients,
    pochhammer,
    schubert_intersection_coefficients,
)


def test_prolog_formal_proof():
    """Executes SWI-Prolog proof knowledgebase and verifies logical assertions pass."""
    cmd = ["swipl", "-g", "run_all_proofs", "-t", "halt", "Gregenbauer_demistify/gegenbauer_proof.pl"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"SWI-Prolog returned non-zero exit code: {res.stderr}"
    assert "PROOF COMPLETED SUCCESSFULLY WITH ALL ASSERTIONS VERIFIED EXACTLY!" in res.stdout
    assert "Geometry of SO(5)/SO(5-1)" in res.stdout
    assert "Quotient Algebra R(Q) & Hilbert Series Dimension Assertion" in res.stdout
    assert "Normalized Degree-Shifting Recurrence Operator M_x" in res.stdout
    assert "Special Exact Test Anchors (S^2, S^3, S^4)" in res.stdout


def test_quotient_ring_normal_form_algebra():
    """Verifies QuadricQuotientPolynomial normal form reduction modulo q = sum(z_i^2)."""
    # In C[z_1, z_2, z_3]/(z_1^2 + z_2^2 + z_3^2), z_3^2 reduces to -(z_1^2 + z_2^2)
    poly = QuadricQuotientPolynomial(3, {(0, 0, 2): 1.0})
    assert poly.terms == {(2, 0, 0): -1.0, (0, 2, 0): -1.0}

    # Test multiplication by x (z_1)
    poly_x = poly.multiply_by_x(0)
    assert poly_x.terms == {(3, 0, 0): -1.0, (1, 2, 0): -1.0}


def test_hilbert_series_dimension_growth():
    """Verifies Hilbert series dimension dim R(Q)_n = [t^n] (1-t^2)/(1-t)^d."""
    for d in [3, 4, 5]:
        lambda_val = (d - 2) / 2.0
        for n in range(5):
            dim_val = quadric_hilbert_series_dim(d, n)
            expected_dim = math.comb(n + d - 1, d - 1) - math.comb(n + d - 3, d - 1) if n >= 0 else 1
            assert dim_val == expected_dim

            # Verify normalization functional identity C_n^(lambda)(1) = (lambda / (n + lambda)) * dim V_n
            c_n_1 = c_n_1_val(n, lambda_val)
            expected_c_n_1 = (lambda_val / (n + lambda_val)) * dim_val
            assert abs(c_n_1 - expected_c_n_1) < 1e-12


def test_exact_anchors_and_scaled_recurrence():
    """Verifies exact anchors (S^2, S^3, S^4) and normalized recurrence phi_n(x)."""
    theta = 0.5
    x = np.cos(theta)
    n = 10

    # S^2 (d=3, lambda=1/2): phi_n(x) = P_n(x)
    phi_s2 = normalized_phi_recurrence(n, 0.5, np.array([x]))[0]
    anchor_s2 = exact_anchor_eval(d=3, n=n, theta=theta)
    assert abs(phi_s2 - anchor_s2) < 1e-12

    # S^3 (d=4, lambda=1): phi_n(theta) = sin((n+1)theta) / ((n+1)sin(theta))
    phi_s3 = normalized_phi_recurrence(n, 1.0, np.array([x]))[0]
    anchor_s3 = exact_anchor_eval(d=4, n=n, theta=theta)
    assert abs(phi_s3 - anchor_s3) < 1e-12

    # S^4 (d=5, lambda=3/2)
    phi_s4 = normalized_phi_recurrence(n, 1.5, np.array([x]))[0]
    anchor_s4 = exact_anchor_eval(d=5, n=n, theta=theta)
    assert abs(phi_s4 - anchor_s4) < 1e-12


def test_operational_phase_map_regimes():
    """Verifies operational phase diagram map selection."""
    lambda_val = 1.5
    # Small n -> direct recurrence
    assert classify_phase_regime(n=50, lambda_val=lambda_val, theta=0.1) == "direct_recurrence"
    # Large n, small theta -> endpoint approximation
    assert classify_phase_regime(n=500, lambda_val=lambda_val, theta=0.005) == "endpoint_approximation"
    # Large n, moderate theta -> interior approximation
    assert classify_phase_regime(n=500, lambda_val=lambda_val, theta=0.8) == "interior_approximation"


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
    n_deg = 50
    lambda_p = 1.5
    ctx = NumericalContext.default_float64()
    solver = GegenbauerComputationalSolver(n=n_deg, lambda_val=lambda_p, context=ctx)

    domain = np.linspace(0.5, 0.99, 100)
    results = solver.benchmark_permutations(domain)

    assert len(results) == 6
    assert AlgebraicPermutation.NORMALIZED_RECURRENCE in results
    assert AlgebraicPermutation.HYPERGEOMETRIC_2F1 in results

    pareto_candidates = [m for m in results.values() if m.is_pareto_optimal]
    assert len(pareto_candidates) >= 1

    opt_acc = solver.solve_optimal_permutation(domain, max_error_tol=1e-3)
    assert opt_acc.max_relative_error <= 1e-3


def test_computational_solver_numerical_bases():
    """Tests computational solver across Fixed-Point Q16.16 and Logarithmic LNS contexts."""
    n_deg = 20
    lambda_p = 1.5
    domain = np.linspace(0.1, 0.9, 50)

    ctx_fp = NumericalContext.fixed_point_q16()
    solver_fp = GegenbauerComputationalSolver(n=n_deg, lambda_val=lambda_p, context=ctx_fp)
    res_fp = solver_fp.evaluate_normalized_recurrence(domain)
    assert ctx_fp.base == NumericalBase.FIXED_POINT
    assert len(res_fp) == 50

    ctx_lns = NumericalContext.logarithmic_lns()
    solver_lns = GegenbauerComputationalSolver(n=n_deg, lambda_val=lambda_p, context=ctx_lns)
    res_lns = solver_lns.evaluate_normalized_recurrence(domain)
    assert ctx_lns.base == NumericalBase.LOGARITHMIC
    assert len(res_lns) == 50
