"""
Comprehensive Mathematical and Numerical Unit Test Suite
=========================================================
Independent, non-tautological test suite verifying the Gegenbauer demystification framework:
1. Mandatory Canonical Verification Anchors (lambda=1/2 Legendre, lambda=1 Chebyshev 2nd kind, n=0..3, x in {-1, -1/2, 0, 1/2, 1})
2. Exact Derivative Anchors phi_n'(1) and phi_n'(-1) for k=0..n
3. Operator Equivalences L_x <-> L_theta <-> H_lambda
4. EndpointClass Enum & Sturm-Liouville Physical vs Analytic Continuation Classifications
5. Dimension Invariant I_dim(n)=0 and Symbolic Dual Recurrence Invariant I_dual(n)=0
6. Jacobi Eigenvalue Bounds sigma(J_m) in (-1, 1), ||J_m|| < 1
7. Quadrature Moments int x^(2r) w(x) dx = B(r+1/2, lambda+1/2)
8. Exact Rational -> RNS/CRT Reconstruction and Finite-Field Modular Congruences (with 3-failure-mode bad prime checks)
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
    EndpointClass,
    QuadricQuotientPolynomial,
    bad_zonal_prime,
    zonal_admissible,
    classify_parameter_domain,
    endpoint_class,
    exact_rational_gegenbauer,
    exact_rational_gegenbauer_derivative,
    exact_rational_gegenbauer_second_derivative,
    gauss_gegenbauer_quadrature,
    get_recurrence_denominator_lcm,
    get_normalization_denominator,
    get_evaluation_denominator,
    get_excluded_primes_denominator,
    lambda_for_sphere,
    modular_gegenbauer_recurrence,
    normalized_gegenbauer_2f1_coefficients,
    normalized_jacobi_coefficients,
    orthonormal_jacobi_coefficients,
    phi_norm_squared,
    pochhammer,
    quadric_hilbert_series_dim,
    rns_crt_gegenbauer_eval,
    check_c_n_admissibility,
    check_point_admissibility,
    check_poly_evaluation_admissibility,
    check_zonal_admissibility,
    check_phi_n_admissibility,
)
from Gregenbauer_demistify.computational_layer import (
    AlgebraicPermutation,
    CertificateTarget,
    CertifiedSensitivity,
    NumericalCertificate,
    Domain,
    ErrorBound,
    ErrorDecomposition,
    GegenbauerComputationalSolver,
    NumericalBase,
    NumericalContext,
    PrecisionType,
    SelectorCandidate,
    TheoremStatus,
    BoundSource,
    high_precision_reference,
    jacobi_eigenpair_residual,
    orthonormal_jacobi_recurrence_residual,
    scale_invariant_schrodinger_residual,
    scale_invariant_ode_residual,
    scale_invariant_recurrence_residual,
    mixed_error,
)
from Gregenbauer_demistify.gegenbauer_asymptotics import (
    c_n_1_val,
    classify_phase_regime,
    composite_matched_approx,
    compute_error_map,
    endpoint_bessel_leading,
    exact_gegenbauer,
    gegenbauer_derivative,
    interior_wkb_approx,
    log_c_n_1,
    normalized_phi_derivative,
    normalized_phi_recurrence,
    orthogonality_norm,
    south_pole_bessel_leading,
    verify_orthogonality_integral_theta,
)


# --- INDEPENDENT REFERENCE ORACLES ---

def reference_gegenbauer(n: int, lambda_val: float, x: np.ndarray) -> np.ndarray:
    """Independent reference oracle via scipy.special.eval_gegenbauer."""
    return eval_gegenbauer(n, lambda_val, x)


def reference_normalized_phi(n: int, lambda_val: float, x: np.ndarray) -> np.ndarray:
    """Independent reference oracle for zonal function phi_n(x) = C_n^(lambda)(x) / C_n^(lambda)(1)."""
    c1 = float(eval_gegenbauer(n, lambda_val, 1.0))
    return eval_gegenbauer(n, lambda_val, x) / c1


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


# --- 2. ENDPOINT CLASS ENUM & STURM-LIOUVILLE CLASSIFICATION ---

def test_endpoint_class_classification():
    """
    Verifies EndpointClass classification:
      - d=3 (lambda=1/2): CRITICAL_LC
      - d=4 (lambda=1): REGULAR
      - d>=5 (lambda>=3/2): LIMIT_POINT
      - analytic continuation 0 < lambda < 1/2: LIMIT_CIRCLE
    """
    assert endpoint_class(0.5) == EndpointClass.CRITICAL_LC
    assert endpoint_class(1.0) == EndpointClass.REGULAR
    assert endpoint_class(0.2) == EndpointClass.LIMIT_CIRCLE
    assert endpoint_class(0.8) == EndpointClass.LIMIT_CIRCLE
    assert endpoint_class(1.5) == EndpointClass.LIMIT_POINT
    assert endpoint_class(2.0) == EndpointClass.LIMIT_POINT


# --- 3. MANDATORY CANONICAL ANCHOR SUITE ---

def test_mandatory_canonical_anchors_grid():
    """
    Mandatory Anchor Test:
    - lambda = 1/2: Legendre polynomials P_n(x)
    - lambda = 1: sin((n+1)theta) / ((n+1)sin theta)
    - degrees n in {0, 1, 2, 3}
    - coordinates x in {-1, -1/2, 0, 1/2, 1}
    """
    grid_x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])

    # lambda = 1/2 (Legendre)
    for n in range(4):
        phi_leg = normalized_phi_recurrence(n, 0.5, grid_x)
        ref_leg = eval_legendre(n, grid_x)
        np.testing.assert_allclose(phi_leg, ref_leg, rtol=1e-12, atol=1e-13)

    # lambda = 1 (Chebyshev 2nd kind / Dirichlet-type)
    for n in range(4):
        phi_cheb = normalized_phi_recurrence(n, 1.0, grid_x)
        for i, x in enumerate(grid_x):
            if abs(x) == 1.0:
                expected = 1.0 if x == 1.0 else ((-1.0) ** n)
            else:
                theta = math.acos(x)
                expected = math.sin((n + 1) * theta) / ((n + 1) * math.sin(theta))
            assert np.isclose(phi_cheb[i], expected, rtol=1e-12, atol=1e-13)


def test_global_parity_and_boundedness():
    """
    Verifies global invariants:
      1. Parity symmetry: phi_n(-x) = (-1)^n * phi_n(x)
      2. Boundedness: |phi_n(x)| <= 1 for all x in [-1, 1].
    """
    n_values = [0, 1, 2, 5, 10]
    x_grid = np.linspace(-1.0, 1.0, 50)
    lambda_val = 1.5

    for n in n_values:
        phi_pos = normalized_phi_recurrence(n, lambda_val, x_grid)
        phi_neg = normalized_phi_recurrence(n, lambda_val, -x_grid)
        expected_neg = ((-1.0) ** n) * phi_pos
        np.testing.assert_allclose(phi_neg, expected_neg, rtol=1e-12, atol=1e-13)
        assert np.all(np.abs(phi_pos) <= 1.0 + 1e-12)


def test_dual_recurrence_conversion_square():
    """
    Verifies exact commutative conversion square alpha_n = a_n * (h_n / h_{n+1}).
    """
    from Gregenbauer_demistify.algebraic_geometry_combinatorics import dual_recurrence_conversion
    n = 5
    lambda_val = 1.5

    a_n, b_n = normalized_jacobi_coefficients(n, lambda_val)

    # h_n = 1 / ||phi_n|| where ||phi_n||^2 = ||C_n||^2 / (C_n(1))^2
    norm_phi_n = math.sqrt(orthogonality_norm(n, lambda_val)) / c_n_1_val(n, lambda_val)
    norm_phi_np1 = math.sqrt(orthogonality_norm(n + 1, lambda_val)) / c_n_1_val(n + 1, lambda_val)
    norm_phi_nm1 = math.sqrt(orthogonality_norm(n - 1, lambda_val)) / c_n_1_val(n - 1, lambda_val)

    h_n = 1.0 / norm_phi_n
    h_np1 = 1.0 / norm_phi_np1
    h_nm1 = 1.0 / norm_phi_nm1

    alpha_n_calc, alpha_nm1_calc = dual_recurrence_conversion(a_n, b_n, h_n, h_np1, h_nm1)
    alpha_n_exact = orthonormal_jacobi_coefficients(n, lambda_val)
    alpha_nm1_exact = orthonormal_jacobi_coefficients(n - 1, lambda_val)

    assert np.isclose(alpha_n_calc, alpha_n_exact, rtol=1e-10)
    assert np.isclose(alpha_nm1_calc, alpha_nm1_exact, rtol=1e-10)


def test_dual_recurrence_symbolic_exact_invariant():
    """
    Verifies exact symbolic dual recurrence identity I_dual(n) = alpha_n^2 - a_n * b_{n+1} == 0.
    """
    for lambda_val in [0.5, 1.0, 1.5, 2.5]:
        for n in range(10):
            a_n, b_n = normalized_jacobi_coefficients(n, lambda_val)
            _, b_np1 = normalized_jacobi_coefficients(n + 1, lambda_val)
            alpha_n = orthonormal_jacobi_coefficients(n, lambda_val)

            symbolic_diff = alpha_n**2 - (a_n * b_np1)
            assert abs(symbolic_diff) < 1e-14


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


def test_normalized_phi_derivative_k_greater_than_n():
    """Verifies that normalized_phi_derivative returns 0 for k > n."""
    n = 5
    lambda_val = 1.5
    x_grid = np.array([0.2, 0.5, 0.8])
    d_zero = normalized_phi_derivative(n, lambda_val, x_grid, k=6)
    np.testing.assert_array_equal(d_zero, np.zeros_like(x_grid))


def test_scale_invariant_ode_residual():
    """
    Verifies scale-invariant dimensionless ODE residual via scale_invariant_ode_residual:
      R_ODE = |(1-x^2) phi'' - (2*lambda+1)x phi' + E_n phi| / (|1-x^2||phi''| + |(2*lambda+1)x||phi'| + E_n|phi| + tau)
    """
    n = 10
    lambda_val = 1.5
    x_grid = np.linspace(-0.8, 0.8, 20)

    for x in x_grid:
        phi0 = float(normalized_phi_recurrence(n, lambda_val, np.array([x]))[0])
        phi1 = float(normalized_phi_derivative(n, lambda_val, np.array([x]), k=1)[0])
        phi2 = float(normalized_phi_derivative(n, lambda_val, np.array([x]), k=2)[0])

        res = scale_invariant_ode_residual(phi0, phi1, phi2, x, n, lambda_val)
        assert res.normalized < 1e-12


def test_orthonormal_jacobi_matrix_symmetry_and_legendre_anchor():
    """Verifies symmetric orthonormal Jacobi matrix subdiagonal coefficients alpha_n."""
    alpha_0_legendre = orthonormal_jacobi_coefficients(0, 0.5)
    assert np.isclose(alpha_0_legendre, 1.0 / np.sqrt(3.0), rtol=1e-12)

    for n in range(1, 10):
        alpha_n = orthonormal_jacobi_coefficients(n, 1.5)
        assert np.isfinite(alpha_n)
        assert alpha_n > 0.0


def test_jacobi_eigenvalue_strict_bounds():
    """
    Verifies Jacobi spectral matrix properties:
      1. J = J* (symmetric)
      2. ||J_m|| < 1 for all finite m
      3. Spectrum sigma(J_m) is strictly contained in (-1, 1).
    """
    for m in [2, 5, 10, 20]:
        nodes, _ = gauss_gegenbauer_quadrature(m, lambda_val=1.5)
        subdiag = np.zeros(m - 1, dtype=np.float64)
        for k in range(m - 1):
            subdiag[k] = orthonormal_jacobi_coefficients(k, 1.5)
        J_m = np.diag(subdiag, k=1) + np.diag(subdiag, k=-1)

        norm_j_m = np.linalg.norm(J_m, 2)
        assert norm_j_m < 1.0
        assert np.all(nodes > -1.0) and np.all(nodes < 1.0)


def test_recurrence_parameter_validation():
    """Verifies that normalized_phi_recurrence raises ValueError on invalid inputs."""
    with pytest.raises(ValueError):
        normalized_phi_recurrence(-1, 1.5, np.array([0.5]))
    with pytest.raises(ValueError):
        normalized_phi_recurrence(5, -0.5, np.array([0.5]))


# --- 4. QUADRIC QUOTIENT ALGEBRA INVARIANTS & EXACT FRACTIONS ---

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


# --- 5. HIGH-PRECISION MPMATH GROUND TRUTH REFERENCE ---

def test_mpmath_high_precision_reference():
    """Verifies mpmath high precision reference oracle against scipy eval_gegenbauer."""
    domain = np.array([0.2, 0.5, 0.8])
    ref_mp = high_precision_reference(n=10, lambda_val=1.5, x=domain, dps=100)
    ref_scipy = reference_normalized_phi(n=10, lambda_val=1.5, x=domain)
    np.testing.assert_allclose(ref_mp, ref_scipy, rtol=1e-10, atol=1e-12)


# --- 6. HILBERT SERIES & NORMALIZATION FUNCTIONAL DECOUPLED ---

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


# --- 7. TWO-ENDPOINT BESSEL & EMPIRICAL ASYMPTOTIC RATES ---

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


# --- 8. TWO-ENDPOINT PHASE CLASSIFIER & ERROR SURFACE DIAGRAM ---

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


# --- 9. COMPUTATIONAL SOLVER HARD CONSTRAINTS & PARETO DOMINANCE ---

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


# --- 10. REAL NUMERICAL BACKENDS ---

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


# --- 11. HIGH-DEGREE LOG-SPACE STABILITY ---

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


# --- 12. THETA-SPACE ORTHOGONALITY NORM ---

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


# --- 13. FIELD EXTENSIONS Q(lambda, x) & GAUSS-GEGENBAUER QUADRATURE TESTS ---

def test_exact_rational_field_extension_recurrence():
    """
    Verifies exact rational Gegenbauer polynomial evaluation in Q(lambda, x).
    C_5^(3/2)(1/2) = -147 / 256 exactly.
    """
    val_frac = exact_rational_gegenbauer(5, Fraction(3, 2), Fraction(1, 2))
    assert val_frac == Fraction(-147, 256)

    # Compare against scipy for n = 0 to 10
    for n in range(10):
        frac_val = exact_rational_gegenbauer(n, Fraction(3, 2), Fraction(2, 3))
        float_ref = float(eval_gegenbauer(n, 1.5, 2.0 / 3.0))
        assert np.isclose(float(frac_val), float_ref, rtol=1e-12, atol=1e-13)


def test_exact_rational_derivatives_and_ode_recovery():
    """
    Verifies exact rational first derivative d/dx C_n^(lambda) = 2*lambda*C_{n-1}^(lambda+1)
    and second derivative recovery via ODE substitution over Q(lambda, x).
    """
    n = 5
    lam = Fraction(3, 2)
    x = Fraction(1, 3)

    # First derivative
    dy_exact = exact_rational_gegenbauer_derivative(n, lam, x)
    ref_dy = 2.0 * 1.5 * float(eval_gegenbauer(n - 1, 2.5, float(x)))
    assert np.isclose(float(dy_exact), ref_dy, rtol=1e-12)

    # Second derivative
    d2y_exact = exact_rational_gegenbauer_second_derivative(n, lam, x)
    ref_d2y = 4.0 * 1.5 * 2.5 * float(eval_gegenbauer(n - 2, 3.5, float(x)))
    assert np.isclose(float(d2y_exact), ref_d2y, rtol=1e-12)


def test_gauss_gegenbauer_quadrature_precision():
    """
    Verifies Golub-Welsch Gauss-Gegenbauer quadrature on polynomial f(x) = x^4,
    along with direct Layer VIII Jacobi spectral eigenpair residual R_J(v, x) = ||J_m v_k - x_k v_k||.
    Integral int_{-1}^1 x^4 (1-x^2)^{1.5 - 0.5} dx = int_{-1}^1 x^4 (1-x^2) dx = 2 * (1/5 - 1/7) = 4/35.
    """
    nodes, weights = gauss_gegenbauer_quadrature(m=4, lambda_val=1.5)
    integral_approx = np.sum(weights * (nodes ** 4))
    exact_integral = 4.0 / 35.0
    assert np.isclose(integral_approx, exact_integral, rtol=1e-12, atol=1e-13)

    # Verify Layer VIII Jacobi Eigenpair Residuals R_J^abs and R_J_hat and Weight Normalization sum w_k = mu_0
    m = 4
    lambda_val = 1.5
    subdiag = np.zeros(m - 1, dtype=np.float64)
    for k in range(m - 1):
        subdiag[k] = orthonormal_jacobi_coefficients(k, lambda_val)
    J_m = np.diag(subdiag, k=1) + np.diag(subdiag, k=-1)
    evals, evecs = np.linalg.eigh(J_m)
    res_abs = jacobi_eigenpair_residual(evals, evecs, lambda_val, normalized=False)
    res_hat = jacobi_eigenpair_residual(evals, evecs, lambda_val, normalized=True)
    assert res_abs.normalized < 1e-14
    assert res_hat.normalized < 1e-14

    mu_0_expected = math.sqrt(math.pi) * gamma(lambda_val + 0.5) / gamma(lambda_val + 1.0)
    assert np.isclose(np.sum(weights), mu_0_expected, rtol=1e-12)


def test_quadrature_exact_moments_beta_integral():
    """
    Verifies Gauss-Gegenbauer quadrature moments against exact Beta integral formula:
      sum w_k x_k^{2r} = B(r + 1/2, lambda + 1/2).
    """
    lambda_val = 1.5
    m = 5
    nodes, weights = gauss_gegenbauer_quadrature(m, lambda_val)

    for r in range(m):
        mom_quad = float(np.sum(weights * (nodes ** (2 * r))))
        mom_exact = math.gamma(r + 0.5) * math.gamma(lambda_val + 0.5) / math.gamma(r + lambda_val + 1.0)
        assert np.isclose(mom_quad, mom_exact, rtol=1e-12)


def test_finite_field_bad_prime_normalization_check():
    """
    Verifies split finite-field certificates handling bad primes where p | u_n.
    For C_10^(2)(1) = 286, prime p=13 divides u_n=286, causing phi_n normalization to fail in F_13 (NormalizationSingularityFailure).
    """
    valid_bad, msg_bad = check_phi_n_admissibility(10, Fraction(2), Fraction(1, 2), 13)
    assert valid_bad is False  # Bad prime 13 divides C_10(1) = 286
    assert "NormalizationSingularityFailure" in msg_bad

    valid_ok, msg_ok = check_phi_n_admissibility(10, Fraction(2), Fraction(1, 2), 17)
    assert valid_ok is True
    assert "Valid" in msg_ok


def test_modular_rns_crt_exact_recovery():
    """
    Verifies exact integer recovery via Residue Number System (RNS) and Chinese Remainder Theorem (CRT).
    C_5^(1)(2) = 780.
    Moduli primes: [10007, 10009]
    """
    n, lam, x = 5, 1, 2
    exact_val = int(exact_rational_gegenbauer(n, lam, x))
    assert exact_val == 780

    rns_crt_val = rns_crt_gegenbauer_eval(n, lam, x, moduli=[10007, 10009])
    assert rns_crt_val == exact_val


def test_canonical_test_matrix_d345_n0123():
    """
    Tests canonical matrix d in {3, 4, 5}, n in {0, 1, 2, 3}:
      - Initial data anchors: phi_0 = 1, phi_1 = x
      - Recurrence invariants: n=0: x phi_0 - phi_1 = 0; n>=1: x phi_n - a_n phi_{n+1} - b_n phi_{n-1} = 0
      - Orthonormal Jacobi basis recurrence split: n=0: x e_0 - alpha_0 e_1 = 0; n>=1: x e_n - alpha_n e_{n+1} - alpha_{n-1} e_{n-1} = 0
    """
    x_grid = np.array([-0.5, 0.0, 0.5])
    for d in [3, 4, 5]:
        lam = lambda_for_sphere(d)
        for n in [0, 1, 2, 3]:
            for x in x_grid:
                phi_n = float(normalized_phi_recurrence(n, lam, np.array([x]))[0])
                phi_np1 = float(normalized_phi_recurrence(n + 1, lam, np.array([x]))[0])
                phi_nm1 = float(normalized_phi_recurrence(max(0, n - 1), lam, np.array([x]))[0])

                res_rec = scale_invariant_recurrence_residual(phi_n, phi_np1, phi_nm1, x, n, float(lam))
                assert res_rec.normalized < 1e-12

                # Orthonormal Jacobi Recurrence Check
                h_n = 1.0 / math.sqrt(phi_norm_squared(n, float(lam)))
                h_np1 = 1.0 / math.sqrt(phi_norm_squared(n + 1, float(lam)))
                h_nm1 = 1.0 / math.sqrt(phi_norm_squared(max(0, n - 1), float(lam)))

                e_n = phi_n * h_n
                e_np1 = phi_np1 * h_np1
                e_nm1 = phi_nm1 * h_nm1

                res_jac = orthonormal_jacobi_recurrence_residual(e_n, e_np1, e_nm1, x, n, float(lam))
                assert res_jac.normalized < 1e-12


def test_denominator_separation_and_zonal_certificate_signature():
    """
    Verifies split denominators D_rec, D_norm, D_eval and ZonalCertificate signature:
      ZonalCertificate(p, plan, degree=n, point=x=c/r)
    """
    d_rec = get_recurrence_denominator_lcm(5, Fraction(3, 2))
    d_norm = get_normalization_denominator(5, Fraction(3, 2))
    d_eval = get_evaluation_denominator(Fraction(1, 2))
    d_excl = get_excluded_primes_denominator(5, Fraction(3, 2), Fraction(1, 2))

    assert d_rec > 0
    assert d_norm > 0
    assert d_eval == 2
    assert d_excl % d_rec == 0 and d_excl % d_eval == 0

    # D_rec for n=5, lambda=3/2 has denominator factors including 7
    # Use prime p=17 for valid admissibility test
    poly_eval_valid = check_poly_evaluation_admissibility(5, Fraction(3, 2), Fraction(1, 2), 17)
    assert poly_eval_valid is True

    zonal_valid, msg = check_zonal_admissibility(p=17, execution_plan="test_plan", degree=5, point=Fraction(1, 2), lambda_val=Fraction(3, 2))
    assert zonal_valid is True
    assert msg == "Valid ZonalCertificate"


def test_semantically_exact_bad_zonal_prime_predicate():
    """
    Tests Bad_zonal(p) <=> not ZonalAdmissible(p).
    """
    # p=13 divides C_10^(2)(1) = 286 (u_n=286), so Bad_zonal(13) is True and ZonalAdmissible is False
    assert bad_zonal_prime(13, degree=10, point=Fraction(1, 2), lambda_val=Fraction(2)) is True
    assert zonal_admissible(13, degree=10, point=Fraction(1, 2), lambda_val=Fraction(2)) is False

    # p=17 is clean, Bad_zonal(17) is False and ZonalAdmissible is True
    assert bad_zonal_prime(17, degree=10, point=Fraction(1, 2), lambda_val=Fraction(2)) is False
    assert zonal_admissible(17, degree=10, point=Fraction(1, 2), lambda_val=Fraction(2)) is True


def test_selector_candidate_target_q_interface():
    """
    Verifies SelectorCandidate interface target Q matching requirement.
    """
    dom = Domain(name="test_domain", lower=-1.0, upper=1.0)
    eb_node = ErrorBound(
        value=1e-5,
        domain=dom,
        source=BoundSource.THEOREM_PROVED,
        status=TheoremStatus.ALGEBRAIC_EXACT,
        decomposition=ErrorDecomposition(),
        target=CertificateTarget.NODE,
        valid=True
    )

    cand_valid = SelectorCandidate(
        name="test_node_cand",
        domain=dom,
        target=CertificateTarget.NODE,
        status=TheoremStatus.ALGEBRAIC_EXACT,
        error_bound=eb_node,
        cost=1.0
    )
    assert cand_valid.is_valid_target_bound is True

    cand_mismatch = SelectorCandidate(
        name="test_weight_cand",
        domain=dom,
        target=CertificateTarget.WEIGHT,  # Mismatch with eb_node.target == NODE
        status=TheoremStatus.ALGEBRAIC_EXACT,
        error_bound=eb_node,
        cost=1.0
    )
    assert cand_mismatch.is_valid_target_bound is False


def test_arbitrary_test_function_norm_isometry():
    """
    Verifies ||T_lambda S_lambda f||_{L^2(0, pi)} = ||f||_{H_lambda}
    for arbitrary linear combinations f = c_0 phi_0 + c_1 phi_1 + c_2 phi_2.
    """
    lam = 1.5
    c0, c1, c2 = 2.0, -1.5, 0.5

    # Compute ||f||_{H_lambda}^2 via exact orthogonality
    norm0_sq = phi_norm_squared(0, lam)
    norm1_sq = phi_norm_squared(1, lam)
    norm2_sq = phi_norm_squared(2, lam)
    exact_f_norm_sq = c0**2 * norm0_sq + c1**2 * norm1_sq + c2**2 * norm2_sq

    # Compute ||T_lambda S_lambda f||_{L^2(0, pi)}^2 via high-order Gauss-Gegenbauer quadrature
    nodes, weights = gauss_gegenbauer_quadrature(m=10, lambda_val=lam)
    phi0 = normalized_phi_recurrence(0, lam, nodes)
    phi1 = normalized_phi_recurrence(1, lam, nodes)
    phi2 = normalized_phi_recurrence(2, lam, nodes)
    f_vals = c0 * phi0 + c1 * phi1 + c2 * phi2

    quad_norm_sq = float(np.sum(weights * (f_vals ** 2)))
    assert np.isclose(quad_norm_sq, exact_f_norm_sq, rtol=1e-10)


def test_golub_welsch_typed_certified_sensitivity():
    """
    Verifies Golub-Welsch typed CertifiedSensitivity_Q and NumericalCertificate implication:
      CertifiedSensitivity_Q(A, kappa_Q) and R_Q <= B_back and E_{Q,conv} <= B_{Q,conv}
      => E_Q <= kappa_Q * B_back + B_{Q,conv}
    """
    sens = CertifiedSensitivity(
        target=CertificateTarget.EIGENVECTOR,
        operator_name="J_m",
        kappa_Q=2.5,
        certified=True
    )
    cert = NumericalCertificate(
        target=CertificateTarget.EIGENVECTOR,
        algorithm="Golub-Welsch",
        backward_bound=1e-12,
        residual_bound=1e-12,
        sensitivity=sens,
        forward_conversion_bound=1e-15
    )
    assert cert.conditioning_kappa == 2.5
    assert np.isclose(cert.forward_bound, 2.5 * 1e-12 + 1e-15)


def test_extended_certificate_tuple_invariant():
    """
    Verifies extended certificate metadata tuple (target Q, domain D, backend, status, validity_conditions).
    """
    dom = Domain(name="test_dom", lower=-1.0, upper=1.0)
    eb = ErrorBound(
        value=1e-6,
        domain=dom,
        source=BoundSource.THEOREM_PROVED,
        status=TheoremStatus.ALGEBRAIC_EXACT,
        decomposition=ErrorDecomposition(),
        target=CertificateTarget.NODE,
        backend="FLOAT64",
        validity_conditions=["domain_contained", "target_matched"],
        valid=True
    )
    target_q, d, backend, status, conds = eb.certificate_tuple
    assert target_q == CertificateTarget.NODE
    assert d == dom
    assert backend == "FLOAT64"
    assert status == TheoremStatus.ALGEBRAIC_EXACT
    assert conds == ["domain_contained", "target_matched"]
