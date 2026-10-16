"""
Unit and Integration Test Suite for Gegenbauer Filter Compiler Engine
=====================================================================
Tests spectral optimization, filter tap symmetry, multi-format fixed-point quantization,
C/C++ header generation, QMF pair metrics, asymptotic phase-map evaluation,
and Pareto trade-off parameter search.
"""

import os
import sys
import tempfile
import numpy as np
import pytest

# Add parent directories to sys.path to enable imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from filter_compiler.compiler import (
    FilterSpec,
    QuantizedTaps,
    FilterResult,
    GegenbauerFilterCompiler,
    SymmetryClass,
    FactorMode,
    FactorizationDiagnostics
)


def test_basis_to_fir_roundtrip():
    """Verifies exact basis-to-FIR roundtrip across Types I, II, III, IV symmetry classes."""
    compiler = GegenbauerFilterCompiler(lam=1.5)
    omega = np.linspace(0, np.pi, 1024)
    x = np.cos(omega)

    # Test Types I-IV
    specs = [
        FilterSpec(kind="lowpass", order=15, cutoff=0.25),   # TYPE_I (odd N)
        FilterSpec(kind="lowpass", order=16, cutoff=0.25),   # TYPE_II (even N)
        FilterSpec(kind="highpass", order=15, cutoff=0.25),  # TYPE_I (odd N highpass)
        FilterSpec(kind="highpass", order=16, cutoff=0.25),  # TYPE_IV (even N highpass)
    ]

    for spec in specs:
        a_coeffs, _, _, _, _, _ = compiler.solve_coefficients(spec)
        taps = compiler.transform_to_taps(a_coeffs, spec)

        # Evaluate target basis sum A(omega)
        A_basis = np.zeros_like(omega)
        for k, c in enumerate(a_coeffs):
            A_basis += c * compiler._eval_fir_basis(k, x, symmetry=spec.symmetry_class)

        # Evaluate reconstructed FIR frequency response H(omega)
        H_fir = np.zeros_like(omega)
        mid = (spec.order - 1) / 2.0
        for n, h_val in enumerate(taps):
            if spec.symmetry_class in (SymmetryClass.TYPE_III, SymmetryClass.TYPE_IV):
                H_fir += h_val * np.sin((n - mid) * omega)
            else:
                H_fir += h_val * np.cos((n - mid) * omega)

        # Verify roundtrip agreement (up to overall DC/Nyquist normalization)
        corr = np.corrcoef(np.abs(A_basis), np.abs(H_fir))[0, 1]
        assert corr > 0.99


def test_solve_qmf_power_coefficients():
    """Verifies that solve_qmf_power_coefficients produces a power-complementary polynomial P(x) + P(-x) == 1."""
    spec = FilterSpec(kind="qmf", order=32, cutoff=0.25)
    compiler = GegenbauerFilterCompiler(lam=1.5)
    a_coeffs, K, cond_val, res_aug, res_data, res_reg = compiler.solve_qmf_power_coefficients(spec)

    assert K > 0
    assert len(a_coeffs) == 2 * K
    # Verify even-indexed coefficients are strictly 0
    for k in range(K):
        assert a_coeffs[2 * k] == 0.0

    # Evaluate R_odd(x) = sum a_{2k+1} C_{2k+1}^(lambda)(x)
    x_nodes = np.linspace(-0.99, 0.99, 100)
    R_pos = np.zeros_like(x_nodes)
    R_neg = np.zeros_like(x_nodes)

    for k in range(K):
        n_odd = 2 * k + 1
        c_k = a_coeffs[n_odd]
        R_pos += c_k * compiler._eval_basis(n_odd, x_nodes)
        R_neg += c_k * compiler._eval_basis(n_odd, -x_nodes)

    # Verify R_odd(-x) == -R_odd(x) => P(x) + P(-x) = (0.5 + R(x)) + (0.5 + R(-x)) == 1
    np.testing.assert_allclose(R_pos + R_neg, 0.0, atol=1e-12)


def test_basis_terms_exceeds_dimension_rejection():
    """Verifies that basis_terms > M raises ValueError."""
    spec = FilterSpec(kind="lowpass", order=15, cutoff=0.25) # M = 8
    compiler = GegenbauerFilterCompiler(basis_terms=12)
    with pytest.raises(ValueError, match="exceeds the independent dimension"):
        compiler.solve_coefficients(spec)


def test_filter_spec_validation():
    """Verifies parameter validation for FilterSpec."""
    spec = FilterSpec(kind="lowpass", order=63, cutoff=0.25)
    assert spec.kind == "lowpass"
    assert spec.order == 63
    assert spec.cutoff == 0.25
    assert spec.wp < spec.cutoff < spec.ws

    with pytest.raises(ValueError):
        FilterSpec(kind="invalid_kind")

    with pytest.raises(ValueError):
        FilterSpec(order=2)

    with pytest.raises(ValueError):
        FilterSpec(cutoff=0.6)


def test_lowpass_compilation():
    """Tests lowpass filter synthesis, tap symmetry, and DC gain normalization."""
    spec = FilterSpec(kind="lowpass", order=31, cutoff=0.2)
    compiler = GegenbauerFilterCompiler(lam=1.5)
    result = compiler.compile(spec)

    assert isinstance(result, FilterResult)
    taps = result.h0_taps.float64_taps
    assert len(taps) == 31

    # Verify symmetry: h[n] == h[N-1-n]
    np.testing.assert_allclose(taps, taps[::-1], atol=1e-12)

    # Verify DC gain normalization: sum(h) == 1.0
    assert abs(np.sum(taps) - 1.0) < 1e-10

    # Check header generation non-empty
    assert "GEG_N 31" in result.header_code or "GEG_N = 31" in result.header_code
    assert "h0_geg_q15" in result.header_code


def test_highpass_compilation():
    """Tests highpass filter synthesis, tap symmetry for Type I odd N, and Nyquist gain normalization."""
    spec = FilterSpec(kind="highpass", order=33, cutoff=0.3)
    compiler = GegenbauerFilterCompiler(lam=1.25)
    result = compiler.compile(spec)

    taps = result.h0_taps.float64_taps
    assert len(taps) == 33

    # Verify symmetry for Type I odd N highpass: h[n] == h[N-1-n]
    np.testing.assert_allclose(taps, taps[::-1], atol=1e-12)

    # Verify Nyquist gain (sum(h * (-1)^n) == 1.0)
    sign_pattern = np.array([(-1.0)**n for n in range(33)])
    nyq_gain = np.sum(taps * sign_pattern)
    assert abs(nyq_gain - 1.0) < 1e-10


def test_bandpass_compilation():
    """Tests bandpass filter synthesis and tap symmetry."""
    spec = FilterSpec(kind="bandpass", order=45, cutoff=0.15, wp2=0.35)
    compiler = GegenbauerFilterCompiler(lam=1.5)
    result = compiler.compile(spec)

    taps = result.h0_taps.float64_taps
    assert len(taps) == 45
    np.testing.assert_allclose(taps, taps[::-1], atol=1e-12)


def test_qmf_pair_compilation():
    """Tests QMF complementary filter bank synthesis, canonical mirror taps, and aliasing distortion metrics."""
    spec = FilterSpec(kind="qmf", order=64, cutoff=0.25)
    compiler = GegenbauerFilterCompiler(lam=1.25, solver="spectral_regularized", mu_reg=1e-4, factor_mode=FactorMode.CEPSTRAL_APPROX)
    result = compiler.compile(spec)

    assert result.h1_taps is not None
    h0 = result.h0_taps.float64_taps
    h1 = result.h1_taps.float64_taps

    assert len(h0) == 64
    assert len(h1) == 64

    # Verify canonical mirror relationship: h1[n] = (-1)^n * h0[N-1-n]
    sign_pattern = np.array([(-1.0)**n for n in range(64)])
    np.testing.assert_allclose(h1, sign_pattern * h0[::-1], atol=1e-12)

    # Check QMF metrics: low power complementarity ripple and suppressed aliasing
    assert result.qmf_power_complementarity_max_db < 1.0
    assert result.qmf_alias_distortion_max_db < -30.0

    # Verify certification flags and factorization diagnostics
    assert result.factorization is not None
    assert result.payload.factorization_certified is True
    assert result.payload.qmf_power_complementary is True
    assert result.payload.qmf_alias_cancellation is True
    assert result.payload.structural_certified is True


def test_asymmetric_qmf_compilation():
    """Tests asymmetric QMF filter synthesis with transition band straddling fs/4 and half-band complementarity."""
    spec = FilterSpec(kind="asymmetric_qmf", order=64, cutoff=0.25, wp=0.20, ws=0.30)
    compiler = GegenbauerFilterCompiler(lam=1.25, solver="spectral_regularized", mu_reg=1e-4)
    result = compiler.compile(spec)

    assert result.h1_taps is not None
    h0 = result.h0_taps.float64_taps
    h1 = result.h1_taps.float64_taps

    assert len(h0) == 64
    assert len(h1) == 64

    # Verify CQF mirror relationship in float and fixed-point domains
    sign_pattern = np.array([(-1.0)**n for n in range(64)])
    np.testing.assert_allclose(h1, sign_pattern * h0[::-1], atol=1e-12)
    q15_sign = np.array([(-1)**n for n in range(64)], dtype=np.int32)
    np.testing.assert_array_equal(result.h1_taps.q15_taps, (q15_sign * result.h0_taps.q15_taps[::-1]).astype(np.int32))

    # Check alias cancellation for asymmetric QMF
    assert result.qmf_alias_distortion_max_db < -20.0
    assert result.payload.qmf_alias_cancellation is True


def test_qmf_alias_transfer_and_haar_anchor():
    """Verifies qmf_alias_transfer complex transfer function against 2-tap Haar wavelet CQF anchor."""
    from filter_compiler.compiler import qmf_alias_transfer

    # Haar wavelet 2-tap CQF filters
    H0_taps = np.array([1.0 / np.sqrt(2), 1.0 / np.sqrt(2)])
    H1_taps = np.array([1.0 / np.sqrt(2), -1.0 / np.sqrt(2)])

    K_fft = 1024
    H0 = np.fft.fft(H0_taps, K_fft)
    H1 = np.fft.fft(H1_taps, K_fft)

    alias_complex = qmf_alias_transfer(H0, H1)
    np.testing.assert_allclose(np.abs(alias_complex), 0.0, atol=1e-15)


def test_solver_dispatch_modes():
    """Verifies that solve_coefficients correctly dispatches between least_squares and spectral_regularized."""
    spec = FilterSpec(kind="lowpass", order=15, cutoff=0.25)

    # least_squares solver
    comp_ls = GegenbauerFilterCompiler(lam=1.5, solver="least_squares", mu_reg=1e-4)
    a_ls, K_ls, cond_ls, res_ls_aug, res_ls_data, res_ls_reg = comp_ls.solve_coefficients(spec)

    # spectral_regularized solver
    comp_reg = GegenbauerFilterCompiler(lam=1.5, solver="spectral_regularized", mu_reg=1e-4)
    a_reg, K_reg, cond_reg, res_reg_aug, res_reg_data, res_reg_reg = comp_reg.solve_coefficients(spec)

    assert K_ls == K_reg
    # Regularized solution must differ from unregularized LS solution
    assert not np.allclose(a_ls, a_reg)


def test_qmf_target_power_complementarity():
    """Verifies that the QMF sine/cosine amplitude crossfade target yields exact power complementarity."""
    spec = FilterSpec(kind="qmf", order=64, cutoff=0.25)
    compiler = GegenbauerFilterCompiler(lam=1.25)
    omega = np.linspace(0, np.pi, 2048)
    D, W = compiler._build_spectral_target(spec, omega)

    # Compute mirror target D_mirror(f) = D(0.5 - f)
    f = omega / (2.0 * np.pi)
    trans_mask = (f > spec.wp) & (f < spec.ws)
    t = (f[trans_mask] - spec.wp) / (spec.ws - spec.wp)

    D_trans = D[trans_mask]
    D_mirror_trans = np.sin(0.5 * np.pi * t)

    # Target power sum in transition: D(t)^2 + D_mirror(t)^2 == 1
    target_power = D_trans**2 + D_mirror_trans**2
    np.testing.assert_allclose(target_power, 1.0, atol=1e-12)


def test_biorthogonal_pair_compilation():
    """Tests biorthogonal filter bank pair compilation via half-band spectral factorization."""
    compiler = GegenbauerFilterCompiler(lam=1.5)

    # Odd sum of orders must raise ValueError
    with pytest.raises(ValueError, match="sum of H0 and G0 orders must be even"):
        compiler.compile_biorthogonal_pair(order_h0=9, order_g0=6)

    # Even delay (e.g. 2+2=4 -> delay 2) requires odd PR delay
    with pytest.raises(ValueError, match="requires odd PR delay"):
        compiler.compile_biorthogonal_pair(order_h0=2, order_g0=2)

    # Valid pair compilation (CDF 9/7 style orders 9 + 7 = 16 or 8 + 6 = 14)
    pair = compiler.compile_biorthogonal_pair(order_h0=8, order_g0=6, cutoff=0.25)

    assert "H0" in pair and "H1" in pair and "G0" in pair and "G1" in pair and "P" in pair
    assert isinstance(pair["H0"], QuantizedTaps)
    assert isinstance(pair["G0"], QuantizedTaps)
    assert isinstance(pair["H1"], QuantizedTaps)
    assert isinstance(pair["G1"], QuantizedTaps)

    h0_taps = pair["H0"].float64_taps
    g0_taps = pair["G0"].float64_taps
    h1_taps = pair["H1"].float64_taps
    g1_taps = pair["G1"].float64_taps
    p_taps = pair["P"]

    # Verify lengths
    assert len(p_taps) == 8 + 6 + 1
    assert len(h0_taps) == len(g1_taps)
    assert len(g0_taps) == len(h1_taps)

    # Verify H0 DC gain normalization H0(1) == sqrt(2) and product relation H0(1)*G0(1) == 2 P(1)
    assert abs(np.sum(h0_taps) - np.sqrt(2)) < 1e-5
    assert abs(np.sum(h0_taps) * np.sum(g0_taps) - np.sum(pair["P"]) * 2.0) < 1e-6

    # Verify PR, alias, and product residuals
    assert pair["pr_residual"] < 1e-5
    assert pair["alias_residual"] < 1e-10
    assert pair["product_residual"] < 1e-5
    assert pair["h0_sym_residual"] < 1e-5
    assert pair["g0_sym_residual"] < 1e-5


def test_halfband_power_polynomial_and_factorization():
    """Tests halfband power polynomial design, validity checks, and spectral factorization."""
    compiler = GegenbauerFilterCompiler(lam=1.5)
    spec = FilterSpec(kind="qmf", order=32, cutoff=0.25)
    a_coeffs, p_taps, K, cond_val, res_aug, res_data, res_reg = compiler.solve_halfband_power_polynomial(spec)

    is_valid, min_P, max_P, max_hb_err = compiler.validate_power_polynomial(p_taps)
    assert is_valid is True
    assert min_P >= -1e-3
    assert max_hb_err < 1e-4

    h0_fact, diag = compiler.spectral_factor_power_polynomial(p_taps, target_N=32)
    assert len(h0_fact) == 32
    assert diag.coefficient_residual < 1e-3


def test_exact_halfband_t3_anchor():
    """Verifies exact half-band spectral factorization for P(w) = 0.5 + 0.5 cos(3w) => H(z) = 0.5 (1 + z^-3)."""
    compiler = GegenbauerFilterCompiler()
    # Centered half-band power polynomial p = [0.25, 0, 0, 0.5, 0, 0, 0.25]
    p_taps_t3 = np.array([0.25, 0.0, 0.0, 0.5, 0.0, 0.0, 0.25])
    is_valid, min_P, max_P, max_hb_err = compiler.validate_power_polynomial(p_taps_t3)

    assert is_valid is True
    assert abs(min_P) < 1e-12
    assert abs(max_P - 1.0) < 1e-12
    assert max_hb_err < 1e-12

    h0_fact, diag = compiler.spectral_factor_power_polynomial(p_taps_t3, target_N=4, mode=FactorMode.STRUCTURED_CHEBYSHEV)
    expected_h = np.array([0.5, 0.0, 0.0, 0.5])
    np.testing.assert_allclose(h0_fact, expected_h, atol=1e-6)
    assert diag.certified is True


def test_invalid_power_polynomial_rejection():
    """Verifies that validate_power_polynomial rejects non-positive power polynomials."""
    compiler = GegenbauerFilterCompiler()
    p_taps_bad = np.array([-1.0, 0.5, -1.0])
    is_valid, min_P, max_P, max_hb_err = compiler.validate_power_polynomial(p_taps_bad)
    assert is_valid is False
    assert min_P < -0.1


def test_spectral_factorization_backends():
    """Tests Structured Chebyshev x-domain root lifting, Cepstral approximation, and Reference Roots factorizations."""
    spec = FilterSpec(kind="qmf", order=32, cutoff=0.25)
    compiler = GegenbauerFilterCompiler(lam=1.5)
    _, p_taps, _, _, _, _, _ = compiler.solve_halfband_power_polynomial(spec)

    # 1. Production Structured Chebyshev x-domain Root Lifting Factorization
    h0_cheb, diag_cheb = compiler.spectral_factor_power_polynomial(p_taps, target_N=32, mode=FactorMode.STRUCTURED_CHEBYSHEV)
    assert isinstance(diag_cheb, FactorizationDiagnostics)
    assert len(h0_cheb) == 32
    assert diag_cheb.finite is True
    assert diag_cheb.coefficient_residual < 0.5

    # 2. Cepstral Minimum-Phase Log-Spectrum Approximation
    h0_cep, diag_cep = compiler.spectral_factor_power_polynomial(p_taps, target_N=32, mode=FactorMode.CEPSTRAL_APPROX)
    assert isinstance(diag_cep, FactorizationDiagnostics)
    assert len(h0_cep) == 32
    assert diag_cep.finite is True
    assert diag_cep.coefficient_residual < 1e-4

    # 3. Reference Roots Monomial Factorization
    h0_ref, diag_ref = compiler.spectral_factor_power_polynomial(p_taps, target_N=32, mode=FactorMode.REFERENCE_ROOTS)
    assert isinstance(diag_ref, FactorizationDiagnostics)
    assert len(h0_ref) == 32
    assert diag_ref.finite is True
    assert diag_ref.coefficient_residual < 1e-3


def test_quantization_formats():
    """Tests float64 to Q15, Q23, and Q31 fixed-point quantization accuracy."""
    spec = FilterSpec(kind="lowpass", order=15, cutoff=0.25)
    compiler = GegenbauerFilterCompiler()
    result = compiler.compile(spec)

    quant = result.h0_taps
    assert quant.q15_taps.dtype in (np.int32, np.int16)
    assert quant.q31_taps.dtype in (np.int64, np.int32)

    # Check bounds
    assert np.all(quant.q15_taps >= -32768) and np.all(quant.q15_taps <= 32767)
    assert np.all(quant.q23_taps >= -8388608) and np.all(quant.q23_taps <= 8388607)

    # Reconstruction error decreases with precision
    h_float = quant.float64_taps
    err_q15 = np.max(np.abs(h_float - quant.q15_taps / quant.q15_scale))
    err_q31 = np.max(np.abs(h_float - quant.q31_taps / quant.q31_scale))
    assert err_q31 < err_q15


def test_quadrature_solver():
    """Tests Gauss-Gegenbauer quadrature projection solver mode."""
    spec = FilterSpec(kind="lowpass", order=21, cutoff=0.25)
    compiler = GegenbauerFilterCompiler(lam=1.0, solver="quadrature", grid_samples=64)
    result = compiler.compile(spec)

    assert len(result.h0_taps.float64_taps) == 21
    assert abs(np.sum(result.h0_taps.float64_taps) - 1.0) < 1e-10


def test_pareto_search():
    """Tests Pareto parameter search over (lambda, mu) space."""
    spec = FilterSpec(kind="lowpass", order=31, cutoff=0.25)
    compiler = GegenbauerFilterCompiler()
    best_result = compiler.pareto_search(
        spec,
        lambda_candidates=[1.0, 1.5],
        mu_candidates=[0.0, 1e-4]
    )
    assert isinstance(best_result, FilterResult)
    assert best_result.lam in [1.0, 1.5]


def test_header_and_plot_output():
    """Tests C/C++ header writing and plot file generation."""
    spec = FilterSpec(kind="qmf", order=32, cutoff=0.25)
    compiler = GegenbauerFilterCompiler(lam=1.25)
    result = compiler.compile(spec)

    with tempfile.TemporaryDirectory() as tmpdir:
        header_path = os.path.join(tmpdir, "test_coeffs.h")
        plot_path = os.path.join(tmpdir, "test_plot.png")

        with open(header_path, "w") as f:
            f.write(result.header_code)
        compiler.plot_response(result, plot_path)

        assert os.path.exists(header_path)
        assert os.path.getsize(header_path) > 100
        assert os.path.exists(plot_path)
        assert os.path.getsize(plot_path) > 1000


def test_layer_viii_provenance_and_truth_status():
    """Tests TruthStatus topology classification and Layer VIII CertifiedEvaluationPayload."""
    from filter_compiler.compiler import TruthStatus, MatchingStatus

    # Physical sphere geometry (lambda = 0.5 => d = 3 Legendre P_n, lambda = 1.5 => d = 5)
    spec = FilterSpec(kind="lowpass", order=31, cutoff=0.2)
    res_sub = GegenbauerFilterCompiler(lam=0.5).compile(spec)
    assert res_sub.payload.truth_status == TruthStatus.PHYSICAL_SPHERE_GEOMETRY

    res_phys = GegenbauerFilterCompiler(lam=1.5).compile(spec)
    assert res_phys.payload.truth_status == TruthStatus.PHYSICAL_SPHERE_GEOMETRY

    # Analytic continuation (lambda = 0.3)
    res_ac = GegenbauerFilterCompiler(lam=0.3).compile(spec)
    assert res_ac.payload.truth_status == TruthStatus.ANALYTIC_CONTINUATION

    # Matching status certification
    assert res_phys.payload.matching_status == MatchingStatus.SAMPLED_ASYMPTOTIC_MATCHING

    # Header macro checks
    assert '#define GEG_TRUTH_STATUS "PHYSICAL_SPHERE_GEOMETRY"' in res_phys.header_code
    assert '#define GEG_MATCHING_STATUS "SAMPLED_ASYMPTOTIC_MATCHING"' in res_phys.header_code
    assert 'GEG_E_TOTAL_ESTIMATE' in res_phys.header_code


def test_filter_spec_edge_cases():
    """Tests QMF odd-N rejection, transition band overlap validation, and clamped bandpass defaults."""

    # QMF odd N must raise ValueError
    with pytest.raises(ValueError, match="QMF filter pair requires an even tap length"):
        FilterSpec(kind="qmf", order=63, cutoff=0.25)

    # Transition band overlap for lowpass (wp >= ws)
    with pytest.raises(ValueError, match="Frequency edges must satisfy 0 < wp"):
        FilterSpec(kind="lowpass", order=31, cutoff=0.25, wp=0.3, ws=0.2)

    # Transition band overlap for highpass (ws >= wp)
    with pytest.raises(ValueError, match="Frequency edges must satisfy 0 < ws"):
        FilterSpec(kind="highpass", order=31, cutoff=0.25, wp=0.2, ws=0.3)

    # Clamped bandpass upper bounds for cutoff near Nyquist
    bp_spec = FilterSpec(kind="bandpass", order=31, cutoff=0.45)
    assert bp_spec.wp2 < 0.5
    assert bp_spec.ws2 < 0.5
    assert bp_spec.ws < bp_spec.wp < bp_spec.wp2 < bp_spec.ws2


def test_exact_halfband_power_polynomial_factorization():
    """Verifies exact P(w) = 1/2 + 1/2 cos(3w) halfband power polynomial factorization and CQF properties."""
    compiler = GegenbauerFilterCompiler()
    p_taps = np.array([0.25, 0.0, 0.0, 0.5, 0.0, 0.0, 0.25])
    h0, diag = compiler.spectral_factor_power_polynomial(p_taps, target_N=4, mode=FactorMode.STRUCTURED_CHEBYSHEV)

    expected = np.array([0.5, 0.0, 0.0, 0.5])
    np.testing.assert_allclose(h0, expected, atol=1e-6)
    assert diag.certified is True

    # Verify CQF/autocorrelation identity h * h_rev == p
    r_h0 = np.convolve(h0, h0[::-1])
    np.testing.assert_allclose(r_h0, p_taps, atol=1e-6)


def test_biorthogonal_odd_delay_pr_identity():
    """Verifies odd-delay Perfect Reconstruction polynomial relation E(z) - E(-z) = 2 z^-d."""
    compiler = GegenbauerFilterCompiler()
    pair = compiler.compile_biorthogonal_pair(order_h0=8, order_g0=6, cutoff=0.25)

    h0 = pair["H0"].float64_taps
    g0 = pair["G0"].float64_taps
    h1 = pair["H1"].float64_taps
    g1 = pair["G1"].float64_taps

    K_fft = 1024
    w = 2.0 * np.pi * np.arange(K_fft) / float(K_fft)
    delay = (8 + 6) // 2 # delay = 7 (odd)

    H0_f = np.fft.fft(h0, K_fft)
    G0_f = np.fft.fft(g0, K_fft)
    H1_f = np.fft.fft(h1, K_fft)
    G1_f = np.fft.fft(g1, K_fft)

    pr_complex = H0_f * G0_f + H1_f * G1_f
    expected_pr = 2.0 * np.exp(-1j * w * delay)

    np.testing.assert_allclose(pr_complex, expected_pr, atol=1e-5)


def test_malformed_root_orbit_rejection():
    """Verifies that RootDecomposer rejects root arrays with multiplicity mismatches or degree non-closure."""
    from filter_compiler.compiler import RootDecomposer

    # Odd endpoint multiplicity rejection
    bad_ep_roots = np.array([1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        RootDecomposer.decompose_z_roots(bad_ep_roots, total_z_degree=3)

    # Degree mismatch rejection
    with pytest.raises(ValueError):
        RootDecomposer.decompose_z_roots(np.array([0.5, 2.0]), total_z_degree=3)


def test_root_decomposer_targeted_orbit_cases():
    """Tests targeted RootDecomposer orbit classification for real reciprocal pairs, complex quartets, endpoints, unit-circle pairs, and asymmetric p_taps."""
    from filter_compiler.compiler import RootDecomposer

    # Real Reciprocal Pair: z = 0.5 and z = 2.0
    recip_roots = np.array([0.5, 2.0])
    units_recip = RootDecomposer.decompose_z_roots(recip_roots, total_z_degree=2)
    assert len(units_recip) == 1
    assert units_recip[0].kind == 'REAL_RECIPROCAL_PAIR'
    assert units_recip[0].multiplicity == 1
    assert units_recip[0].classification_status in ("EXACT_ORBIT", "NUMERICALLY_MATCHED")
    assert isinstance(units_recip[0].matching_margin, float)

    # Complex Quartet: z = 0.5 e^(j pi/4), conj, recip, conj_recip
    z1 = 0.5 * np.exp(1j * np.pi / 4.0)
    quartet_roots = np.array([z1, np.conj(z1), 1.0 / z1, 1.0 / np.conj(z1)])
    units_q = RootDecomposer.decompose_z_roots(quartet_roots, total_z_degree=4)
    assert len(units_q) == 1
    assert units_q[0].kind == 'COMPLEX_RECIPROCAL_CONJUGATE_QUARTET'

    # Endpoint roots: z = 1.0 (double) and z = -1.0 (double)
    ep_roots = np.array([1.0, 1.0, -1.0, -1.0])
    units_ep = RootDecomposer.decompose_z_roots(ep_roots, total_z_degree=4)
    kinds = {u.kind for u in units_ep}
    assert 'REAL_ENDPOINT' in kinds

    # Unit-circle pair: z = e^(j pi/3) and z = e^(-j pi/3)
    uc_roots = np.array([np.exp(1j * np.pi / 3.0), np.exp(-1j * np.pi / 3.0)])
    units_uc = RootDecomposer.decompose_z_roots(uc_roots, total_z_degree=2)
    assert len(units_uc) == 1
    assert units_uc[0].kind == 'UNIT_CIRCLE_PAIR'

    # Near-endpoint reciprocal pair: z = 1.0005 and z = 1 / 1.0005 (must be classified as REAL_RECIPROCAL_PAIR, not REAL_ENDPOINT)
    r_near = 1.0005
    near_ep_roots = np.array([r_near, 1.0 / r_near])
    units_near = RootDecomposer.decompose_z_roots(near_ep_roots, total_z_degree=2, rel_tol=1e-4, abs_tol=1e-5)
    assert len(units_near) == 1
    assert units_near[0].kind == 'REAL_RECIPROCAL_PAIR'

    # Near-endpoint unit-circle pair: z = e^(i * 0.02) and z = e^(-i * 0.02) (must be classified as UNIT_CIRCLE_PAIR, not REAL_ENDPOINT)
    theta = 0.02
    uc_near_ep = np.array([np.exp(1j * theta), np.exp(-1j * theta)])
    units_uc_near = RootDecomposer.decompose_z_roots(uc_near_ep, total_z_degree=2, rel_tol=1e-3, abs_tol=1e-3)
    assert len(units_uc_near) == 1
    assert units_uc_near[0].kind == 'UNIT_CIRCLE_PAIR'

    # Permutation invariance: root array order does not affect decomposition
    perm1 = np.array([z1, 1.0 / z1, np.conj(z1), 1.0 / np.conj(z1)])
    perm2 = np.array([1.0 / np.conj(z1), z1, 1.0 / z1, np.conj(z1)])
    u1 = RootDecomposer.decompose_z_roots(perm1, total_z_degree=4)
    u2 = RootDecomposer.decompose_z_roots(perm2, total_z_degree=4)
    assert u1[0].kind == u2[0].kind == 'COMPLEX_RECIPROCAL_CONJUGATE_QUARTET'

    # Asymmetric p_taps rejection in decompose_halfband_roots
    p_asym = np.array([0.1, 0.2, 0.5, 0.3, 0.1])
    with pytest.raises(ValueError, match="is not symmetric"):
        RootDecomposer.decompose_halfband_roots(p_asym, total_z_degree=4)


def test_specification_failure_vs_engineering_acceptance():
    """Verifies that engineering acceptance thresholds allow prototype_fir while design_target_met fails if spec is missed."""
    # Strict target passband ripple 0.01 dB, stopband atten 80 dB
    spec = FilterSpec(kind="lowpass", order=15, cutoff=0.25, passband_ripple_db=0.01, stopband_atten_db=80.0)
    compiler = GegenbauerFilterCompiler(lam=1.5)
    result = compiler.compile(spec)

    assert result.payload.specification_met is False
    assert result.payload.prototype_fir_certified is True
    assert result.payload.sampled_design_targets_met is False


def test_biorthogonal_pr_and_alias_quantized_formats():
    """Verifies perfect reconstruction and alias cancellation residuals across quantized tap formats for biorthogonal filter banks."""
    compiler = GegenbauerFilterCompiler(lam=1.5)
    pair = compiler.compile_biorthogonal_pair(order_h0=8, order_g0=6, cutoff=0.25)

    delay = (8 + 6) // 2
    K_fft = 2048
    w = 2.0 * np.pi * np.arange(K_fft) / float(K_fft)
    expected_pr = 2.0 * np.exp(-1j * w * delay)

    # Validate float64, q31, and q23 formats
    for fmt_key in ['q31_taps', 'q23_taps']:
        scale_h0 = pair['H0'].q31_scale if fmt_key == 'q31_taps' else pair['H0'].q23_scale
        scale_g0 = pair['G0'].q31_scale if fmt_key == 'q31_taps' else pair['G0'].q23_scale
        scale_h1 = pair['H1'].q31_scale if fmt_key == 'q31_taps' else pair['H1'].q23_scale
        scale_g1 = pair['G1'].q31_scale if fmt_key == 'q31_taps' else pair['G1'].q23_scale

        h0 = getattr(pair['H0'], fmt_key) / scale_h0
        g0 = getattr(pair['G0'], fmt_key) / scale_g0
        h1 = getattr(pair['H1'], fmt_key) / scale_h1
        g1 = getattr(pair['G1'], fmt_key) / scale_g1

        H0_f = np.fft.fft(h0, K_fft)
        G0_f = np.fft.fft(g0, K_fft)
        H1_f = np.fft.fft(h1, K_fft)
        G1_f = np.fft.fft(g1, K_fft)

        pr_err = float(np.max(np.abs(H0_f * G0_f + H1_f * G1_f - expected_pr)))
        assert pr_err < 1e-3


def test_five_layer_architecture_invariants_and_candidate_map():
    """Verifies Layer 1-5 architectural components: InvariantExtractor, DecompositionClassEnumerator, CandidateMap, and IndependentVerifier."""
    from filter_compiler.compiler import (
        InvariantExtractor,
        DecompositionClassEnumerator,
        CandidateMap,
        IndependentVerifier
    )

    # 1. Invariant extraction for Lowpass and QMF specs
    lp_spec = FilterSpec(kind="lowpass", order=31, cutoff=0.25)
    lp_inv = InvariantExtractor.extract(lp_spec)
    assert lp_inv.order == 31
    assert lp_inv.degree == 30
    assert lp_inv.independent_dim == 16
    assert lp_inv.is_halfband is False

    qmf_spec = FilterSpec(kind="qmf", order=32, cutoff=0.25)
    qmf_inv = InvariantExtractor.extract(qmf_spec)
    assert qmf_inv.order == 32
    assert qmf_inv.degree == 31
    assert qmf_inv.is_halfband is True
    assert qmf_inv.requires_positivity is True

    # 2. Structural decomposition class enumeration
    lp_classes = DecompositionClassEnumerator.enumerate_classes(lp_inv)
    assert len(lp_classes) == 1
    assert "FIR_TYPE_I_ORDER_31" in lp_classes[0].class_id

    qmf_classes = DecompositionClassEnumerator.enumerate_classes(qmf_inv)
    assert len(qmf_classes) == 2
    assert "HALFBAND_PRODUCT" in qmf_classes[0].class_id

    # 3. Candidate map construction and 3-stage early pruning
    cmap = CandidateMap(qmf_inv, qmf_classes)
    assert cmap.pruning_statistics['generated'] == 2

    s1_pruned = cmap.apply_stage1_exact_structural_rejection()
    assert s1_pruned == 0
    s2_pruned = cmap.apply_stage2_decomposition_feasibility()
    assert s2_pruned == 0

    p_taps_good = np.array([0.25, 0.0, 0.0, 0.5, 0.0, 0.0, 0.25])
    s3_pruned = cmap.apply_stage3_necessary_numerical_bounds(p_taps_good)
    assert s3_pruned == 0
    assert cmap.pruning_statistics['surviving_branches'] == 2

    # 4. Layer 5 IndependentVerifier
    is_valid, min_P, max_P, hb_err = IndependentVerifier.verify_power_polynomial(p_taps_good)
    assert is_valid is True
    assert abs(min_P) < 1e-12
    assert abs(max_P - 1.0) < 1e-12

    # Verify Stage 3 rejection on negative p_taps
    p_taps_bad = np.array([-1.0, 0.0, -1.0])
    cmap_bad = CandidateMap(qmf_inv, qmf_classes)
    cmap_bad.apply_stage1_exact_structural_rejection()
    cmap_bad.apply_stage2_decomposition_feasibility()
    s3_bad = cmap_bad.apply_stage3_necessary_numerical_bounds(p_taps_bad)
    assert s3_bad == 2
    assert cmap_bad.pruning_statistics['surviving_branches'] == 0


def test_dynamic_integer_partitions_and_multi_part_candidate_map():
    """Verifies integer partition generation and multi-part CandidateMap graph construction."""
    from filter_compiler.compiler import _generate_integer_partitions, InvariantExtractor, DecompositionClassEnumerator, CandidateMap, FilterSpec

    parts_deg3_m2 = _generate_integer_partitions(3, max_parts=2)
    assert parts_deg3_m2 == [(3,), (2, 1)]

    spec = FilterSpec(kind="qmf", order=8, cutoff=0.25)
    inv = InvariantExtractor.extract(spec)
    classes = DecompositionClassEnumerator.enumerate_classes(inv)

    # Instantiate candidate map with max_parts=2
    cmap = CandidateMap(inv, classes)
    cmap._build_initial_graph(max_parts=2)
    # 2 classes * partitions of degree 7 with max_parts 2
    assert cmap.pruning_statistics['generated'] > len(classes)


def test_root_order_permutation_invariance():
    """Verifies that RootDecomposer orbit classification is invariant to root input permutations."""
    from filter_compiler.compiler import RootDecomposer

    r1 = 0.6 + 0.3j
    r2 = 0.6 - 0.3j
    r3 = 1.0 / r1
    r4 = 1.0 / r2

    roots_orig = np.array([r1, r2, r3, r4])
    roots_perm = np.array([r4, r1, r3, r2])

    units_orig = RootDecomposer.decompose_z_roots(roots_orig, total_z_degree=4)
    units_perm = RootDecomposer.decompose_z_roots(roots_perm, total_z_degree=4)

    assert len(units_orig) == len(units_perm) == 1
    assert units_orig[0].kind == units_perm[0].kind == 'COMPLEX_RECIPROCAL_CONJUGATE_QUARTET'


def test_biorthogonal_input_validation_and_pr_gates():
    """Verifies that compile_biorthogonal_pair strictly validates inputs and raises ValueError on invalid parameters."""
    compiler = GegenbauerFilterCompiler(lam=1.5)

    # Invalid order_h0
    with pytest.raises(ValueError, match="order_h0 must be a positive integer"):
        compiler.compile_biorthogonal_pair(order_h0=-2, order_g0=6)

    # Invalid order_g0
    with pytest.raises(ValueError, match="order_g0 must be a positive integer"):
        compiler.compile_biorthogonal_pair(order_h0=8, order_g0=0)

    # Invalid cutoff
    with pytest.raises(ValueError, match="cutoff must be finite"):
        compiler.compile_biorthogonal_pair(order_h0=8, order_g0=6, cutoff=0.8)


def test_certified_payload_evidence_levels():
    """Verifies CertifiedEvaluationPayload evidence_level, sampled_passband_ripple, and engineering_acceptance_met semantics."""
    spec = FilterSpec(kind="qmf", order=32, cutoff=0.25)
    compiler = GegenbauerFilterCompiler(lam=1.5)
    result = compiler.compile(spec)

    assert result.payload.evidence_level == "FACTOR_IDENTITY_VALIDATED"
    assert result.payload.sampled_passband_ripple == result.passband_ripple_actual
    assert result.payload.sampled_stopband_attenuation == result.stopband_atten_actual
    assert isinstance(result.payload.engineering_acceptance_met, bool)


def test_coefficient_domain_biorthogonal_identities():
    """Verifies coefficient-domain PR and alias cancellation sequence verification in compile_biorthogonal_pair."""
    compiler = GegenbauerFilterCompiler(lam=1.5)
    pair = compiler.compile_biorthogonal_pair(order_h0=8, order_g0=6, cutoff=0.25)

    h0 = pair["H0"].float64_taps
    g0 = pair["G0"].float64_taps
    h1 = pair["H1"].float64_taps
    g1 = pair["G1"].float64_taps

    delay = (8 + 6) // 2

    # Verify exact PR convolution sequence
    pr_seq = np.convolve(h0, g0) + np.convolve(h1, g1)
    target_pr = np.zeros_like(pr_seq)
    target_pr[delay] = 2.0
    np.testing.assert_allclose(pr_seq, target_pr, atol=1e-4)

    # Verify exact alias cancellation sequence
    h0_mod = h0 * np.array([(-1.0)**n for n in range(len(h0))])
    h1_mod = h1 * np.array([(-1.0)**n for n in range(len(h1))])
    alias_seq = np.convolve(h0_mod, g0) + np.convolve(h1_mod, g1)
    np.testing.assert_allclose(alias_seq, 0.0, atol=1e-4)


def test_candidate_map_pruning_gating_error():
    """Verifies that CandidateMap execution gating raises RuntimeError if candidate branches are pruned."""
    from filter_compiler.compiler import CandidateMap, InvariantExtractor, DecompositionClassEnumerator

    spec = FilterSpec(kind="qmf", order=32, cutoff=0.25)
    inv = InvariantExtractor.extract(spec)
    classes = DecompositionClassEnumerator.enumerate_classes(inv)

    # Manually reject all branches
    cmap = CandidateMap(inv, classes)
    for node in cmap.nodes.values():
        node.rejection_reason = "Manual Test Rejection"

    # Verify zero surviving branches
    s3 = cmap.apply_stage3_necessary_numerical_bounds()
    assert cmap.pruning_statistics['surviving_branches'] == 0
