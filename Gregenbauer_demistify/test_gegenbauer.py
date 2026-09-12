"""
Comprehensive Mathematical and Numerical Unit Test Suite
=========================================================
Independent, non-tautological test suite verifying the Gegenbauer demystification framework:
1. Independent Reference Oracles (scipy.special.eval_legendre, eval_gegenbauer, mpmath)
2. Decoupled Hilbert Series Dimensions and Normalization Identity
3. Quadric Quotient Ring Normal Forms, Idempotency, Exact Fractions & Invariants Modulo q
4. Symmetric Orthonormal Jacobi Matrix Coefficients alpha_n (alpha_0 = 1/sqrt(3) for Legendre)
5. High-Precision Ground Truth Reference via mpmath (100+ bits)
6. Two-Endpoint Bessel Layer Tests (North & South Poles)
7. Empirical Asymptotic Convergence Exponents (WKB p > 0.9, Bessel p > 1.7)
8. Robust Mixed Error Computational Layer, Hard Constraints & Deterministic Pareto Dominance
9. Real Execution Backends (FLOAT32, FLOAT64, LONGDOUBLE, MPMATH, Q16.16, LNS)
10. High-Degree Log-Space Stability up to n = 10^6
11. Theta-Space Orthogonality Norm Verification (Closed-Form Gamma vs Quadrature)
12. Exact Derivative Anchors phi_n'(1) and phi_n'(-1)
13. Prolog Integration Test with shutil.which and pathlib Resolution
"""

from fractions import Fraction
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy.special import eval_gegenbauer, eval_legendre, gamma, gammaln

from Gregenbauer_demistify.algebraic_geometry_combinatorics import (
    QuadricQuotientPolynomial,
    lambda_for_sphere,
    normalized_gegenbauer_2f1_coefficients,
    normalized_jacobi_coefficients,
    orthonormal_jacobi_coefficients,
    pochhammer,
    quadric_hilbert_series_dim,
)
from Gregenbauer_demistify.computational_layer import (
    AlgebraicPermutation,
    GegenbauerComputationalSolver,
    NumericalBase,
    NumericalContext,
    PrecisionType,
    high_precision_reference,
    mixed_error,
)
from Gregenbauer_demistify.gegenbauer_asymptotics import (
    c_n_1_val,
    classify_phase_regime,
    composite_matched_approx,
    compute_error_map,
    endpoint_bessel_leading,
    exact_gegenbauer,
    interior_wkb_approx,
    log_c_n_1,
    normalized_phi_recurrence,
    orthogonality_norm,
    south_pole_bessel_leading,
    verify_orthogonality_integral_theta,
)


# --- INDEPENDENT REFERENCE ORACLES ---

def reference_gegenbauer(n: int, lambda_val: float, x: np.ndarray) -> np.ndarray:
    """Independent reference oracle via scipy.special.eval_gegenbauer."""
    return np.asarray(eval_gegenbauer(n, lambda_val, x), dtype=np.float64)


def reference_normalized_phi(n: int, lambda_val: float, x: np.ndarray) -> np.ndarray:
    """Independent reference oracle for zonal function phi_n(x) = C_n^(lambda)(x) / C_n^(lambda)(1)."""
    c1 = float(eval_gegenbauer(n, lambda_val, 1.0))
    return np.asarray(eval_gegenbauer(n, lambda_val, x), dtype=np.float64) / c1


# --- 1. PROLOG PROOF INTEGRATION TEST ---

def test_prolog_formal_proof():
    """Robust integration test executing SWI-Prolog proof knowledgebase via pathlib resolution."""
    swipl_bin = shutil.which("swipl") or "/usr/bin/swipl"
    if not Path(swipl_bin).exists():
        pytest.skip("swipl (SWI-Prolog) executable not found")

    root_dir = Path(__file__).resolve().parent
    proof_file = root_dir / "gegenbauer_proof.pl"

    cmd = [swipl_bin, "-g", "run_all_proofs", "-t", "halt", str(proof_file)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"SWI-Prolog returned non-zero exit code: {res.stderr}"
    assert "ALL REGISTERED EXECUTABLE CONSISTENCY CHECKS PASSED SUCCESSFULLY!" in res.stdout


# --- 2. INDEPENDENT ANCHOR TESTS & DERIVATIVES ---

def test_s2_anchor_against_independent_legendre():
    """
    S^2 Anchor (d=3, lambda=0.5): Verifies normalized_phi_recurrence against
    scipy's independent eval_legendre oracle across multiple degrees.
    """
    theta = 0.5
    x = np.cos(theta)
    for n in [0, 1, 2, 10, 50, 100]:
        got = normalized_phi_recurrence(n, 0.5, np.array([x]))[0]
        ref = float(eval_legendre(n, x))
        assert np.isclose(got, ref, rtol=1e-12, atol=1e-13)


def test_s3_anchor_against_independent_trig_formula():
    """
    S^3 Anchor (d=4, lambda=1.0): Verifies normalized_phi_recurrence against
    the explicit trigonometric formula sin((n+1)theta) / ((n+1)sin(theta)).
    """
    theta = 0.4
    x = np.cos(theta)
    for n in [0, 1, 5, 20, 50]:
        got = normalized_phi_recurrence(n, 1.0, np.array([x]))[0]
        ref = np.sin((n + 1.0) * theta) / ((n + 1.0) * np.sin(theta))
        assert np.isclose(got, ref, rtol=1e-12, atol=1e-13)


def test_s4_anchor_against_independent_scipy_ratio():
    """
    S^4 Anchor (d=5, lambda=1.5): Verifies normalized_phi_recurrence against
    independent scipy eval_gegenbauer ratio.
    """
    x = 0.7
    for n in [0, 1, 4, 15, 30]:
        got = normalized_phi_recurrence(n, 1.5, np.array([x]))[0]
        ref = reference_normalized_phi(n, 1.5, np.array([x]))[0]
        assert np.isclose(got, ref, rtol=1e-12, atol=1e-13)


def test_exact_derivative_anchors():
    """
    Verifies exact derivative anchor identities:
      phi_n'(1) = n(n + 2*lambda) / (2*lambda + 1)
      phi_n'(-1) = (-1)^{n-1} * n(n + 2*lambda) / (2*lambda + 1)
    """
    h = 1e-7
    for n in [1, 2, 5, 10]:
        for lambda_val in [0.5, 1.0, 1.5]:
            # North pole x=1
            phi_1 = normalized_phi_recurrence(n, lambda_val, np.array([1.0]))[0]
            phi_1_h = normalized_phi_recurrence(n, lambda_val, np.array([1.0 - h]))[0]
            num_deriv_1 = (phi_1 - phi_1_h) / h
            exact_deriv_1 = (n * (n + 2.0 * lambda_val)) / (2.0 * lambda_val + 1.0)
            assert np.isclose(num_deriv_1, exact_deriv_1, rtol=1e-4)

            # South pole x=-1
            phi_m1 = normalized_phi_recurrence(n, lambda_val, np.array([-1.0]))[0]
            phi_m1_h = normalized_phi_recurrence(n, lambda_val, np.array([-1.0 + h]))[0]
            num_deriv_m1 = (phi_m1_h - phi_m1) / h
            exact_deriv_m1 = ((-1.0) ** (n - 1)) * (n * (n + 2.0 * lambda_val)) / (2.0 * lambda_val + 1.0)
            assert np.isclose(num_deriv_m1, exact_deriv_m1, rtol=1e-4)


def test_orthonormal_jacobi_matrix_symmetry_and_legendre_anchor():
    """Verifies symmetric orthonormal Jacobi matrix subdiagonal coefficients alpha_n."""
    # Sanity check for lambda = 0.5 (Legendre): alpha_0 = 1 / sqrt(3)
    alpha_0_legendre = orthonormal_jacobi_coefficients(0, 0.5)
    assert np.isclose(alpha_0_legendre, 1.0 / np.sqrt(3.0), rtol=1e-12)

    for n in range(1, 10):
        alpha_n = orthonormal_jacobi_coefficients(n, 1.5)
        assert np.isfinite(alpha_n)
        assert alpha_n > 0.0


def test_recurrence_parameter_validation():
    """Verifies that normalized_phi_recurrence raises ValueError on invalid inputs."""
    with pytest.raises(ValueError):
        normalized_phi_recurrence(-1, 1.5, np.array([0.5]))
    with pytest.raises(ValueError):
        normalized_phi_recurrence(5, -0.5, np.array([0.5]))


# --- 3. QUADRIC QUOTIENT ALGEBRA INVARIANTS & EXACT FRACTIONS ---

def test_quadric_exact_fraction_algebra():
    """Verifies exact Fraction arithmetic in QuadricQuotientPolynomial."""
    poly = QuadricQuotientPolynomial(3, {(0, 0, 2): Fraction(1, 2)})
    assert poly.terms == {(2, 0, 0): Fraction(-1, 2), (0, 2, 0): Fraction(-1, 2)}
    assert "-1/2" in repr(poly) and "z" in repr(poly)


def test_quadric_quadric_relation_is_zero():
    """Verifies q(z) = z_1^2 + ... + z_d^2 == 0 in R(Q)."""
    q_poly = QuadricQuotientPolynomial(3, {(2, 0, 0): 1, (0, 2, 0): 1, (0, 0, 2): 1})
    assert q_poly.terms == {}  # Exactly 0 polynomial in quotient ring R(Q)


def test_quadric_z_d_4_multinomial_reduction():
    """Verifies multinomial non-recursive expansion for z_d^4 -> (z_1^2 + z_2^2)^2."""
    poly_z3_4 = QuadricQuotientPolynomial(3, {(0, 0, 4): 1})
    expected = {(4, 0, 0): Fraction(1), (2, 2, 0): Fraction(2), (0, 4, 0): Fraction(1)}
    assert poly_z3_4.terms == expected


def test_lambda_for_sphere_and_exact_jacobi_fractions():
    """Verifies lambda_for_sphere and exact Fraction Jacobi recurrence sum a_n + b_n = 1."""
    assert lambda_for_sphere(5) == Fraction(3, 2)
    assert lambda_for_sphere(4) == Fraction(1)

    for n in range(1, 10):
        a_n, b_n = normalized_jacobi_coefficients(n, Fraction(3, 2), exact=True)
        assert a_n + b_n == Fraction(1)


def test_quotient_ring_zonal_polynomial_projection_accuracy():
    """
    P0 Fix Verification: Verifies that evaluate_quotient_ring_normal_form evaluates
    the genuine spherical zonal polynomial phi_n(x), NOT raw x^n.
    For n=2, lambda=1.5: phi_2(x) = (5x^2 - 1)/4.
    """
    solver = GegenbauerComputationalSolver(n=2, lambda_val=1.5)
    domain = np.array([0.0, 0.5, 0.8, 1.0])
    got = solver.evaluate_quotient_ring_normal_form(domain)
    expected = (5.0 * domain**2 - 1.0) / 4.0
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12)


# --- 4. HIGH-PRECISION MPMATH GROUND TRUTH REFERENCE ---

def test_mpmath_high_precision_reference():
    """Verifies mpmath high precision reference oracle against scipy eval_gegenbauer."""
    domain = np.array([0.2, 0.5, 0.8])
    ref_mp = high_precision_reference(n=10, lambda_val=1.5, x=domain, dps=100)
    ref_scipy = reference_normalized_phi(n=10, lambda_val=1.5, x=domain)
    np.testing.assert_allclose(ref_mp, ref_scipy, rtol=1e-10, atol=1e-12)


# --- 5. HILBERT SERIES & NORMALIZATION FUNCTIONAL DECOUPLED ---

def test_quadric_hilbert_series_dimension_decoupled():
    """Tests Hilbert series dimension formula dim R(Q)_n = binom(n+d-1,d-1) - binom(n+d-3,d-1)."""
    for d in [3, 4, 5, 6]:
        for n in range(6):
            dim_val = quadric_hilbert_series_dim(d, n)
            expected = math.comb(n + d - 1, d - 1) - math.comb(n + d - 3, d - 1)
            assert dim_val == expected


def test_normalization_functional_identity_decoupled():
    """
    Independently verifies the representation functional identity C_n^(lambda)(1) = (lambda/(n+lambda)) * dim V_n
    comparing scipy eval_gegenbauer against closed-form dimension formula.
    """
    for d in [3, 4, 5]:
        lambda_val = (d - 2) / 2.0
        for n in range(1, 10):
            dim_v_n = math.comb(n + d - 1, d - 1) - math.comb(n + d - 3, d - 1)
            c_n_1_scipy = float(eval_gegenbauer(n, lambda_val, 1.0))
            identity_val = (lambda_val / (n + lambda_val)) * dim_v_n
            assert np.isclose(c_n_1_scipy, identity_val, rtol=1e-12)


# --- 6. TWO-ENDPOINT BESSEL & EMPIRICAL ASYMPTOTIC RATES ---

def test_south_pole_bessel_boundary_layer():
    """Verifies South pole Bessel layer phi_n(theta) ~ (-1)^n Cal_J_{lambda-1/2}(K*(pi-theta))."""
    n = 100
    lambda_val = 1.5
    theta_south = np.pi - 0.01

    ref_south = reference_normalized_phi(n, lambda_val, np.array([np.cos(theta_south)]))[0]
    bessel_south = south_pole_bessel_leading(n, lambda_val, np.array([theta_south]))[0]

    assert abs(ref_south - bessel_south) < 1e-3


def test_wkb_empirical_convergence_exponent():
    """
    Calculates empirical convergence exponent p = -log(e1/e0) / log(n1/n0) for WKB
    and asserts p > 0.9 (exact O(1/n) rate p = 1.0).
    """
    lambda_val = 1.5
    theta_val = np.pi / 4.0
    x_val = np.cos(theta_val)
    n_values = [100, 200, 400, 800]
    errors = []

    for n in n_values:
        ref = reference_normalized_phi(n, lambda_val, np.array([x_val]))[0]
        wkb = interior_wkb_approx(n, lambda_val, np.array([theta_val]))[0]
        rel_err = abs(ref - wkb) / abs(ref)
        errors.append(rel_err)

    rates = []
    for i in range(len(n_values) - 1):
        n0, n1 = n_values[i], n_values[i + 1]
        e0, e1 = errors[i], errors[i + 1]
        rate = -np.log(e1 / e0) / np.log(n1 / n0)
        rates.append(rate)

    assert all(r > 0.9 for r in rates)


def test_mehler_heine_empirical_convergence_exponent():
    """
    Calculates empirical convergence exponent p = -log(e1/e0) / log(n1/n0) for Mehler-Heine
    and asserts p > 1.7 (exact O(1/n^2) rate p = 2.0).
    """
    lambda_val = 1.5
    z_fix = 2.5
    n_values = [50, 100, 200, 400]
    errors = []

    for n in n_values:
        K = n + lambda_val
        theta_end = z_fix / K
        x_end = np.cos(theta_end)

        ref_ratio = reference_normalized_phi(n, lambda_val, np.array([x_end]))[0]
        bessel_ratio = endpoint_bessel_leading(n, lambda_val, np.array([theta_end]))[0]

        abs_err = abs(ref_ratio - bessel_ratio)
        errors.append(abs_err)

    rates = []
    for i in range(len(n_values) - 1):
        n0, n1 = n_values[i], n_values[i + 1]
        e0, e1 = errors[i], errors[i + 1]
        rate = -np.log(e1 / e0) / np.log(n1 / n0)
        rates.append(rate)

    assert all(r > 1.7 for r in rates)


# --- 7. TWO-ENDPOINT PHASE CLASSIFIER & ERROR SURFACE DIAGRAM ---

def test_two_endpoint_phase_classifier():
    """
    Verifies two-endpoint phase map classifier across z_0 and z_pi coordinates.
    """
    n = 200
    lambda_val = 1.5

    assert classify_phase_regime(n, lambda_val, 0.01) == "north_endpoint_bessel"
    assert classify_phase_regime(n, lambda_val, np.pi - 0.01) == "south_endpoint_bessel"
    assert classify_phase_regime(n, lambda_val, 1.5) == "interior_approximation"


def test_error_surface_diagram_computation():
    """Tests compute_error_map generation across [0, pi] theta domain."""
    err_map = compute_error_map(n=100, lambda_val=1.5, num_theta=50)
    assert "theta" in err_map
    assert "exact_phi" in err_map
    assert "max_err_wkb" in err_map
    assert np.isfinite(err_map["max_err_wkb"])
    assert np.isfinite(err_map["max_err_composite"])


# --- 8. COMPUTATIONAL SOLVER HARD CONSTRAINTS & PARETO DOMINANCE ---

def test_deterministic_pareto_dominance_logic():
    """Verifies Pareto non-dominance algorithm using synthetic deterministic cost/error metrics."""
    ctx = NumericalContext.default_float64()
    solver = GegenbauerComputationalSolver(n=10, lambda_val=1.5, context=ctx)

    metrics = {
        AlgebraicPermutation.NORMALIZED_RECURRENCE: solver.benchmark_permutations(np.array([0.5]))[AlgebraicPermutation.NORMALIZED_RECURRENCE],
        AlgebraicPermutation.HYPERGEOMETRIC_2F1: solver.benchmark_permutations(np.array([0.5]))[AlgebraicPermutation.HYPERGEOMETRIC_2F1],
    }

    metrics[AlgebraicPermutation.NORMALIZED_RECURRENCE].exec_time_sec = 0.002
    metrics[AlgebraicPermutation.NORMALIZED_RECURRENCE].max_mixed_error = 1e-5

    metrics[AlgebraicPermutation.HYPERGEOMETRIC_2F1].exec_time_sec = 0.001
    metrics[AlgebraicPermutation.HYPERGEOMETRIC_2F1].max_mixed_error = 1e-3

    solver._compute_pareto_frontier(metrics)
    assert metrics[AlgebraicPermutation.NORMALIZED_RECURRENCE].is_pareto_optimal is True
    assert metrics[AlgebraicPermutation.HYPERGEOMETRIC_2F1].is_pareto_optimal is True


def test_solver_hard_constraint_failure():
    """Verifies that solve_optimal_permutation raises ValueError when tolerance is infeasible."""
    solver = GegenbauerComputationalSolver(n=50, lambda_val=1.5)
    domain = np.linspace(-0.8, 0.8, 50)
    with pytest.raises(ValueError):
        solver.solve_optimal_permutation(domain, max_error_tol=1e-30)


# --- 9. REAL NUMERICAL BACKENDS ---

def test_real_numerical_backends():
    """Tests execution across real backends: FLOAT32, FLOAT64, LONGDOUBLE, FIXED_POINT, LNS."""
    n_deg = 20
    lambda_p = 1.5
    domain = np.linspace(0.1, 0.9, 20)

    for prec in [PrecisionType.FLOAT32, PrecisionType.FLOAT64, PrecisionType.LONGDOUBLE]:
        ctx = NumericalContext(precision=prec)
        solver = GegenbauerComputationalSolver(n=n_deg, lambda_val=lambda_p, context=ctx)
        res = solver.evaluate_normalized_recurrence(domain)
        assert np.all(np.isfinite(res))
        assert len(res) == 20

    # Fixed Point
    ctx_fp = NumericalContext.fixed_point_q16()
    solver_fp = GegenbauerComputationalSolver(n=n_deg, lambda_val=lambda_p, context=ctx_fp)
    res_fp = solver_fp.evaluate_normalized_recurrence(domain)
    assert np.all(np.isfinite(res_fp))

    # LNS
    ctx_lns = NumericalContext.logarithmic_lns()
    solver_lns = GegenbauerComputationalSolver(n=n_deg, lambda_val=lambda_p, context=ctx_lns)
    res_lns = solver_lns.evaluate_normalized_recurrence(domain)
    assert np.all(np.isfinite(res_lns))


# --- 10. HIGH-DEGREE LOG-SPACE STABILITY ---

def test_high_degree_log_space_stability():
    """
    Tests log_c_n_1 up to n = 10^6 against independent gammaln formula
    log C_n^(lambda)(1) = gammaln(n+2*lambda) - gammaln(n+1) - gammaln(2*lambda).
    """
    lambda_val = 1.5
    for n in [10**2, 10**3, 10**4, 10**5, 10**6]:
        got_log = log_c_n_1(n, lambda_val)
        ref_log = float(gammaln(n + 2.0 * lambda_val) - gammaln(n + 1.0) - gammaln(2.0 * lambda_val))
        assert np.isfinite(got_log)
        assert np.isclose(got_log, ref_log, rtol=1e-12)


# --- 11. THETA-SPACE ORTHOGONALITY NORM ---

def test_theta_space_orthogonality_norm_quadrature():
    """
    Verifies orthogonality_norm(n, lambda) against theta-space numerical quadrature
    without fractional power singularities: int_0^pi C_n(cos(theta))^2 sin^{2*lambda}(theta) d_theta.
    """
    lambda_val = 1.5
    for n in range(1, 10):
        got_hn = orthogonality_norm(n, lambda_val)
        ref_hn = (np.pi * (2.0 ** (1.0 - 2.0 * lambda_val)) * gamma(n + 2.0 * lambda_val) /
                  (gamma(n + 1.0) * (n + lambda_val) * (gamma(lambda_val) ** 2)))
        assert np.isclose(got_hn, ref_hn, rtol=1e-12)

        # Theta-space Quadrature integration
        num_hn = verify_orthogonality_integral_theta(n, n, lambda_val)
        assert np.isclose(got_hn, num_hn, rtol=1e-3)
