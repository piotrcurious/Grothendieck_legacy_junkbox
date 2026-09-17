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
    GegenbauerFilterCompiler
)


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
    """Tests highpass filter synthesis, tap symmetry, and Nyquist gain normalization."""
    spec = FilterSpec(kind="highpass", order=33, cutoff=0.3)
    compiler = GegenbauerFilterCompiler(lam=1.25)
    result = compiler.compile(spec)

    taps = result.h0_taps.float64_taps
    assert len(taps) == 33

    # Verify symmetry
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
    compiler = GegenbauerFilterCompiler(lam=1.25, solver="spectral_regularized", mu_reg=1e-4)
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

    # Verify certification flags
    assert result.payload.qmf_power_complementary is True
    assert result.payload.qmf_alias_cancellation is True
    assert result.payload.is_certified is True


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

    # Verify DC gain normalization (approx sqrt(2))
    assert abs(np.sum(h0_taps) - np.sqrt(2)) < 1e-5
    assert abs(np.sum(g0_taps) - np.sqrt(2)) < 1e-5


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

    # Subcritical limit circle
    spec = FilterSpec(kind="lowpass", order=31, cutoff=0.2)
    res_sub = GegenbauerFilterCompiler(lam=0.5).compile(spec)
    assert res_sub.payload.truth_status == TruthStatus.LIMIT_CIRCLE_SUBCRITICAL

    # Physical sphere geometry (lambda = 1.5 => d = 5)
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
    assert 'GEG_E_TOTAL_BOUND' in res_phys.header_code


def test_filter_spec_edge_cases():
    """Tests QMF odd-N rejection, transition band overlap validation, and clamped bandpass defaults."""

    # QMF odd N must raise ValueError
    with pytest.raises(ValueError, match="QMF filter pair requires an even tap length"):
        FilterSpec(kind="qmf", order=63, cutoff=0.25)

    # Transition band overlap for lowpass (wp >= ws)
    with pytest.raises(ValueError, match="Passband edge wp .* must be < stopband edge ws"):
        FilterSpec(kind="lowpass", order=31, cutoff=0.25, wp=0.3, ws=0.2)

    # Transition band overlap for highpass (ws >= wp)
    with pytest.raises(ValueError, match="Stopband edge ws .* must be < passband edge wp"):
        FilterSpec(kind="highpass", order=31, cutoff=0.25, wp=0.2, ws=0.3)

    # Clamped bandpass upper bounds for cutoff near Nyquist
    bp_spec = FilterSpec(kind="bandpass", order=31, cutoff=0.45)
    assert bp_spec.wp2 < 0.5
    assert bp_spec.ws2 < 0.5
    assert bp_spec.ws < bp_spec.wp < bp_spec.wp2 < bp_spec.ws2
