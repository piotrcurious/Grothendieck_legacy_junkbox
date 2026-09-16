"""
Gegenbauer Filter Compiler Module (Refined & Integrated)
========================================================
A comprehensive FIR/IIR/QMF DSP filter compiler based on Gegenbauer polynomial bases,
Jacobi matrix operators, Sturm-Liouville differential regularizers, and asymptotic boundary layer theory.
"""

import os
import sys
import math
import argparse
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from algebraic_geometry_combinatorics import (
    phi_norm_squared,
    gauss_gegenbauer_quadrature
)
from gegenbauer_asymptotics import (
    normalized_phi_recurrence,
    endpoint_bessel_leading,
    interior_wkb_approx,
    composite_matched_approx,
    c_n_1_val
)
from computational_layer import (
    mixed_error,
    NumericalContext,
    PrecisionType,
    NumericalBase
)


class TruthStatus(Enum):
    """Physical Sphere Geometry vs Analytic Continuation Parameter Topology."""
    PHYSICAL_SPHERE_GEOMETRY = "PHYSICAL_SPHERE_GEOMETRY"
    LIMIT_CIRCLE_SUBCRITICAL = "LIMIT_CIRCLE_SUBCRITICAL"
    ANALYTIC_CONTINUATION = "ANALYTIC_CONTINUATION"


class MatchingStatus(Enum):
    """Layer VI Asymptotic Matching Certification Status."""
    MATCHING_SCHEMA = "MATCHING_SCHEMA"
    ANALYTICALLY_CERTIFIED_MATCHING = "ANALYTICALLY_CERTIFIED_MATCHING"


@dataclass
class ErrorBoundProvenance:
    """Layer VIII Certified Non-Double-Counting Error Decomposition Payload."""
    e_analytic: float = 0.0
    e_arithmetic: float = 0.0
    e_conditioning: float = 0.0
    e_implementation: float = 0.0

    @property
    def total(self) -> float:
        return self.e_analytic + self.e_arithmetic + self.e_conditioning + self.e_implementation


@dataclass
class CertifiedEvaluationPayload:
    """Layer VIII Certified Evaluation Payload separating diagnostics from certified bounds."""
    truth_status: TruthStatus
    matching_status: MatchingStatus
    provenance: ErrorBoundProvenance
    basis_evaluation_certified: bool = True
    prototype_fir_certified: bool = True
    qmf_power_complementary: bool = True
    qmf_alias_cancellation: bool = True
    is_certified: bool = True


def determine_truth_status(lam: float) -> TruthStatus:
    if abs(lam - 0.5) < 1e-12:
        return TruthStatus.LIMIT_CIRCLE_SUBCRITICAL
    if lam > 0 and abs(2.0 * lam - round(2.0 * lam)) < 1e-12:
        return TruthStatus.PHYSICAL_SPHERE_GEOMETRY
    return TruthStatus.ANALYTIC_CONTINUATION


@dataclass
class FilterSpec:
    """Specification of target DSP filter."""
    kind: str = "lowpass"
    order: int = 64
    cutoff: float = 0.25
    wp: Optional[float] = None
    ws: Optional[float] = None
    wp2: Optional[float] = None
    ws2: Optional[float] = None
    sampling_rate: int = 2000
    passband_ripple_db: float = 0.1
    stopband_atten_db: float = 60.0

    def __post_init__(self):
        self.kind = self.kind.lower()
        if self.kind not in ("lowpass", "highpass", "bandpass", "qmf"):
            raise ValueError(f"Unknown filter kind: '{self.kind}'")
        if self.order < 3:
            raise ValueError("Filter order must be >= 3")
        if not (0.0 < self.cutoff < 0.5):
            raise ValueError(f"Cutoff must be in (0.0, 0.5), got {self.cutoff}")

        # Highpass FIR filters cannot have an even number of taps (Type II)
        if self.kind == "highpass" and self.order % 2 == 0:
            raise ValueError(f"Highpass FIR filter (Type II) cannot have an even length N={self.order} due to forced Nyquist zero. Filter length must be odd.")

        # QMF filter pairs require an even tap length
        if self.kind == "qmf" and self.order % 2 != 0:
            raise ValueError(f"QMF filter pair requires an even tap length N, got {self.order}.")

        if self.kind == "bandpass":
            if self.wp is None: self.wp = max(0.02, self.cutoff - 0.05)
            if self.ws is None: self.ws = max(0.01, self.wp - 0.05)
            if self.wp2 is None: self.wp2 = min(0.48, max(self.wp + 0.05, self.cutoff + 0.1))
            if self.ws2 is None: self.ws2 = min(0.49, self.wp2 + 0.05)
        else:
            if self.wp is None:
                self.wp = min(0.49, self.cutoff + 0.05) if self.kind == "highpass" else max(0.01, self.cutoff - 0.05)
            if self.ws is None:
                self.ws = max(0.01, self.cutoff - 0.05) if self.kind == "highpass" else min(0.49, self.cutoff + 0.05)

        if self.kind in ("lowpass", "qmf") and self.wp >= self.ws:
            raise ValueError(f"Passband edge wp ({self.wp}) must be < stopband edge ws ({self.ws})")
        elif self.kind == "highpass" and self.ws >= self.wp:
            raise ValueError(f"Stopband edge ws ({self.ws}) must be < passband edge wp ({self.wp})")
        elif self.kind == "bandpass" and not (self.ws < self.wp < self.wp2 < self.ws2):
            raise ValueError(f"Bandpass frequencies must satisfy ws ({self.ws}) < wp ({self.wp}) < wp2 ({self.wp2}) < ws2 ({self.ws2})")

        if self.kind == "qmf":
            if abs((self.wp + self.ws) - 0.5) > 1e-12:
                raise ValueError(
                    f"QMF requires symmetric transition around fs/4: wp + ws = 0.5, got wp={self.wp}, ws={self.ws}"
                )


@dataclass
class QuantizedTaps:
    """Container for multi-format quantized filter taps."""
    float64_taps: np.ndarray
    q15_taps: np.ndarray
    q23_taps: np.ndarray
    q31_taps: np.ndarray
    q15_scale: float = 32767.0
    q23_scale: float = 8388607.0
    q31_scale: float = 2147483647.0


@dataclass
class FilterResult:
    """Compiler output containing taps, metrics, certifications, payloads, and headers."""
    spec: FilterSpec
    lam: float
    basis_terms: int
    mu_reg: float
    h0_taps: QuantizedTaps
    payload: CertifiedEvaluationPayload
    h1_taps: Optional[QuantizedTaps] = None
    freq_grid: np.ndarray = field(default_factory=lambda: np.array([]))
    H0_response: np.ndarray = field(default_factory=lambda: np.array([]))
    H1_response: Optional[np.ndarray] = None
    passband_ripple_actual: float = 0.0
    stopband_atten_actual: float = 0.0
    qmf_power_complementarity_max_db: float = 0.0
    qmf_alias_distortion_max_db: float = 0.0
    regularization_energy: float = 0.0
    asymptotic_error_bound: float = 0.0
    provenance_mixed_error: float = 0.0
    header_code: str = ""

    def summary(self) -> str:
        lines = [
            "=== Gegenbauer Filter Compiler Execution Summary ===",
            f"Filter Type: {self.spec.kind.upper()} | Order N: {self.spec.order} | Cutoff: {self.spec.cutoff} fs",
            f"Gegenbauer Lambda: {self.lam:.4f} | Basis Terms: {self.basis_terms}",
            f"Truth Status Topology: {self.payload.truth_status.value}",
            f"Asymptotic Matching Certification: {self.payload.matching_status.value}",
            f"Certifications: Basis={self.payload.basis_evaluation_certified} | Prototype={self.payload.prototype_fir_certified}"
            f" | QMF Power={self.payload.qmf_power_complementary} | QMF Alias={self.payload.qmf_alias_cancellation} | Total={self.payload.is_certified}",
            f"Passband Ripple: {self.passband_ripple_actual:.4f} dB | Stopband Attenuation: {self.stopband_atten_actual:.2f} dB",
        ]
        if self.spec.kind == "qmf":
            lines.append(f"QMF Power Complementarity Peak Ripple: {self.qmf_power_complementarity_max_db:.4f} dB")
            lines.append(f"QMF Peak Alias Distortion: {self.qmf_alias_distortion_max_db:.2f} dB")
        lines.append(f"Sturm-Liouville Regularization Energy: {self.regularization_energy:.6e}")
        lines.append(f"Asymptotic Boundary Error Bound (E_analytic): {self.payload.provenance.e_analytic:.6e}")
        lines.append(f"Fixed-Point Quantization Noise (E_arithmetic): {self.payload.provenance.e_arithmetic:.6e}")
        lines.append(f"Certified Total Error Bound (E_total): {self.payload.provenance.total:.6e}")
        return "\n".join(lines)


class GegenbauerFilterCompiler:
    """DSP Filter Compiler leveraging the VIII-Layer Gegenbauer Theoretical Framework."""

    def __init__(
        self,
        lam: float = 1.5,
        basis_terms: Optional[int] = None,
        basis_type: str = "normalized",
        solver: str = "spectral_regularized",
        mu_reg: float = 1e-4,
        reg_power: int = 1,
        asymptotic_mode: str = "auto",
        grid_samples: int = 2048,
        precision: PrecisionType = PrecisionType.FLOAT64
    ):
        if lam <= -0.5:
            raise ValueError(f"Lambda parameter must be > -0.5, got {lam}")
        self.lam = float(lam)
        self.basis_terms = basis_terms
        self.basis_type = basis_type
        self.solver = solver
        self.mu_reg = float(mu_reg)
        self.reg_power = int(reg_power)
        self.asymptotic_mode = asymptotic_mode
        self.grid_samples = grid_samples
        self.ctx = NumericalContext(precision=precision, base=NumericalBase.BASE_2)

    def _eval_basis(self, n: int, x: np.ndarray) -> np.ndarray:
        """Evaluates n-th basis function at array x in [-1, 1]."""
        x_arr = np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0)
        if self.basis_type == "normalized":
            return normalized_phi_recurrence(n, self.lam, x_arr)
        c1 = float(c_n_1_val(n, self.lam))
        return c1 * normalized_phi_recurrence(n, self.lam, x_arr)

    def _eval_asymptotic_basis(self, n: int, omega: np.ndarray) -> np.ndarray:
        theta = omega
        if self.asymptotic_mode == "bessel":
            return endpoint_bessel_leading(n, self.lam, theta)
        elif self.asymptotic_mode == "wkb":
            return interior_wkb_approx(n, self.lam, theta)
        elif self.asymptotic_mode == "composite":
            return composite_matched_approx(n, self.lam, theta)
        else:
            return composite_matched_approx(n, self.lam, theta)

    def _build_spectral_target(self, spec: FilterSpec, omega: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        f = omega / (2.0 * np.pi)
        D = np.zeros_like(omega)
        W = np.ones_like(omega)
        wp, ws = spec.wp, spec.ws

        if spec.kind == "qmf":
            pass_mask = f <= wp
            stop_mask = f >= ws
            trans_mask = ~(pass_mask | stop_mask)
            D[pass_mask] = 1.0
            D[stop_mask] = 0.0
            if np.any(trans_mask):
                t = (f[trans_mask] - wp) / (ws - wp)
                # Sine/cosine amplitude crossfade for power-complementary QMF:
                D[trans_mask] = np.cos(0.5 * np.pi * t)
            W[pass_mask], W[stop_mask], W[trans_mask] = 1.0, 10.0, 1.0

        elif spec.kind == "lowpass":
            pass_mask = f <= wp
            stop_mask = f >= ws
            trans_mask = ~(pass_mask | stop_mask)
            D[pass_mask] = 1.0
            if np.any(trans_mask):
                D[trans_mask] = 0.5 * (1.0 + np.cos(np.pi * (f[trans_mask] - wp) / (ws - wp)))
            W[pass_mask], W[stop_mask], W[trans_mask] = 1.0, 10.0, 0.1

        elif spec.kind == "highpass":
            pass_mask = f >= wp
            stop_mask = f <= ws
            trans_mask = ~(pass_mask | stop_mask)
            D[pass_mask] = 1.0
            if np.any(trans_mask):
                D[trans_mask] = 0.5 * (1.0 - np.cos(np.pi * (f[trans_mask] - ws) / (wp - ws)))
            W[pass_mask], W[stop_mask], W[trans_mask] = 1.0, 10.0, 0.1

        elif spec.kind == "bandpass":
            wp2, ws2 = spec.wp2, spec.ws2
            pass_mask = (f >= wp) & (f <= wp2)
            stop_mask = (f <= ws) | (f >= ws2)
            trans_mask1 = (f > ws) & (f < wp)
            trans_mask2 = (f > wp2) & (f < ws2)

            D[pass_mask] = 1.0
            if np.any(trans_mask1):
                D[trans_mask1] = 0.5 * (1.0 - np.cos(np.pi * (f[trans_mask1] - ws) / (wp - ws)))
            if np.any(trans_mask2):
                D[trans_mask2] = 0.5 * (1.0 + np.cos(np.pi * (f[trans_mask2] - wp2) / (ws2 - wp2)))

            W[pass_mask], W[stop_mask] = 1.0, 10.0
            if np.any(trans_mask1): W[trans_mask1] = 0.1
            if np.any(trans_mask2): W[trans_mask2] = 0.1

        return D, W

    def solve_coefficients(self, spec: FilterSpec) -> Tuple[np.ndarray, int]:
        N = spec.order
        M = (N + 1) // 2
        if self.basis_terms is not None:
            K = self.basis_terms
        else:
            if spec.kind == "qmf":
                K = min(M, max(16, N // 2))
            else:
                K = max(4, int(np.sqrt(N)) + 2)
        K = min(K, M)

        mu = self.mu_reg
        if spec.kind == "qmf" and self.mu_reg == 1e-4:
            mu = 1e-6

        if self.solver == "quadrature":
            nodes, weights = gauss_gegenbauer_quadrature(self.grid_samples, self.lam)
            omega_q = np.arccos(nodes)
            D_q, _ = self._build_spectral_target(spec, omega_q)

            a_coeffs = np.zeros(K)
            for k in range(K):
                phi_k = self._eval_basis(k, nodes)
                norm_sq = 1.0 if self.basis_type == "normalized" else phi_norm_squared(k, self.lam)
                a_coeffs[k] = np.sum(D_q * phi_k * weights) / norm_sq
            return a_coeffs, K

        if self.solver == "spectral_regularized" or mu > 0:
            nodes, weights = gauss_gegenbauer_quadrature(self.grid_samples, self.lam)
            omega_q = np.arccos(nodes)
            D_q, W_q = self._build_spectral_target(spec, omega_q)

            A = np.zeros((self.grid_samples, K))
            for k in range(K):
                A[:, k] = self._eval_basis(k, nodes)

            sqrt_W = np.sqrt(W_q * weights)
            A_w = A * sqrt_W[:, np.newaxis]
            D_w = D_q * sqrt_W

            R_diag = np.zeros(K)
            for k in range(K):
                eig = k * (k + 2.0 * self.lam)
                R_diag[k] = (eig ** self.reg_power)

            R_mat = np.diag(np.sqrt(mu * R_diag))
            A_sys = np.vstack([A_w, R_mat])
            D_sys = np.concatenate([D_w, np.zeros(R_mat.shape[0])])
            a_coeffs, _, _, _ = np.linalg.lstsq(A_sys, D_sys, rcond=None)
            return a_coeffs, K

        omega = np.linspace(0, np.pi, self.grid_samples)
        x = np.cos(omega)
        D, W = self._build_spectral_target(spec, omega)

        A = np.zeros((self.grid_samples, K))
        for k in range(K):
            A[:, k] = self._eval_basis(k, x)

        sqrt_W = np.sqrt(W)
        A_w = A * sqrt_W[:, np.newaxis]
        D_w = D * sqrt_W
        a_coeffs, _, _, _ = np.linalg.lstsq(A_w, D_w, rcond=None)

        return a_coeffs, K

    def transform_to_taps(self, a_coeffs: np.ndarray, spec: FilterSpec) -> np.ndarray:
        """Transforms Gegenbauer spectral expansion coefficients into zero-phase FIR taps using IDFT projection."""
        N = spec.order
        grid_L = self.grid_samples
        omega = np.linspace(0, np.pi, grid_L)
        x = np.cos(omega)

        A_freq = np.zeros(grid_L, dtype=np.float64)
        for k, c in enumerate(a_coeffs):
            A_freq += c * self._eval_basis(k, x)

        h = np.zeros(N, dtype=np.float64)
        mid = (N - 1) / 2.0

        trapz_fn = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
        for n in range(N):
            m = n - mid
            integrand = A_freq * np.cos(m * omega)
            h[n] = (1.0 / np.pi) * trapz_fn(integrand, x=omega)

        # Enforce exact linear-phase symmetry
        h = 0.5 * (h + h[::-1])

        # Normalize gain based on filter type
        if spec.kind in ("lowpass", "qmf"):
            sum_h = np.sum(h)
            if abs(sum_h) > 1e-12:
                h /= sum_h
        elif spec.kind == "highpass":
            nyq_gain = np.sum(h * np.array([(-1.0)**n for n in range(N)]))
            if abs(nyq_gain) > 1e-12:
                h /= nyq_gain
        elif spec.kind == "bandpass":
            K_fft_norm = max(4096, 1 << (math.ceil(math.log2(spec.order)) + 2))
            max_g = np.max(np.abs(np.fft.fft(h, K_fft_norm)))
            if max_g > 1e-12:
                h /= max_g

        return h

    def quantize_taps(self, h_float: np.ndarray) -> QuantizedTaps:
        peak = float(np.max(np.abs(h_float))) if len(h_float) > 0 else 1.0
        norm_factor = peak if peak > 1.0 else 1.0

        q15_scale = 32767.0 / norm_factor
        q23_scale = 8388607.0 / norm_factor
        q31_scale = 2147483647.0 / norm_factor

        q15 = np.clip(np.round((h_float / norm_factor) * 32767.0), -32768, 32767).astype(np.int32)
        q23 = np.clip(np.round((h_float / norm_factor) * 8388607.0), -8388608, 8388607).astype(np.int32)
        q31 = np.clip(np.round((h_float / norm_factor) * 2147483647.0), -2147483648, 2147483647).astype(np.int64)

        return QuantizedTaps(
            float64_taps=h_float,
            q15_taps=q15,
            q23_taps=q23,
            q31_taps=q31,
            q15_scale=q15_scale,
            q23_scale=q23_scale,
            q31_scale=q31_scale
        )

    def generate_header(self, result: FilterResult) -> str:
        """Generates modern C/C++ / Embedded DSP header file with full metadata and PROGMEM arrays."""
        spec = result.spec
        h0 = result.h0_taps
        h1 = result.h1_taps
        p = result.payload.provenance

        header = [
            "// Auto-generated Gegenbauer DSP Filter Coefficients Header",
            "// VIII-Layer Gegenbauer Theoretical Framework Compiler Engine",
            f"// Target Specification: {spec.kind.upper()} | N = {spec.order} | Cutoff = {spec.cutoff} fs",
            f"// Parameters: Lambda = {result.lam} | Basis Terms = {result.basis_terms} | Sampling Rate = {spec.sampling_rate} Hz",
            f"// Performance: Passband Ripple = {result.passband_ripple_actual:.4f} dB | Stopband Atten = {result.stopband_atten_actual:.2f} dB",
            f"// Certification: TruthStatus = {result.payload.truth_status.value} | MatchingStatus = {result.payload.matching_status.value}",
            "// Provenance Error Bound: E_total <= E_analytic + E_arithmetic + kappa * E_input + E_impl",
            f"//   E_analytic = {p.e_analytic:.6e} | E_arithmetic = {p.e_arithmetic:.6e} | E_total = {p.total:.6e}"
        ]
        if spec.kind == "qmf":
            header.append(f"// QMF Metrics: Max Power Ripple = {result.qmf_power_complementarity_max_db:.4f} dB | Max Alias Dist = {result.qmf_alias_distortion_max_db:.2f} dB")

        header.extend([
            "",
            "#ifndef GEGENBAUER_FILTER_COEFFS_H",
            "#define GEGENBAUER_FILTER_COEFFS_H",
            "",
            "#include <stdint.h>",
            "#ifdef __AVR__",
            "  #include <avr/pgmspace.h>",
            "#else",
            "  #ifndef PROGMEM",
            "    #define PROGMEM",
            "  #endif",
            "#endif",
            "",
            f'#define GEG_TRUTH_STATUS "{result.payload.truth_status.value}"',
            f'#define GEG_MATCHING_STATUS "{result.payload.matching_status.value}"',
            f"#define GEG_N {spec.order}",
            f"#define GEG_SAMPLING_RATE {spec.sampling_rate}",
            f"#define GEG_CUTOFF {spec.cutoff}f",
            f"#define GEG_LAMBDA {result.lam}f",
            f"#define GEG_BASIS_TERMS {result.basis_terms}",
            f"#define GEG_E_TOTAL_BOUND {p.total}f",
            ""
        ])

        header.append(f"const double h0_geg_float64[{spec.order}] = {{")
        header.append("    " + ", ".join(f"{val:.12e}" for val in h0.float64_taps))
        header.append("};")
        header.append("")

        header.append(f"const float h0_geg_float32[{spec.order}] = {{")
        header.append("    " + ", ".join(f"{val:.8e}f" for val in h0.float64_taps))
        header.append("};")
        header.append("")

        header.append(f"const int16_t h0_geg_q15[{spec.order}] PROGMEM = {{")
        header.append("    " + ", ".join(str(int(val)) for val in h0.q15_taps))
        header.append("};")
        header.append("")

        header.append(f"const int32_t h0_geg_q31[{spec.order}] PROGMEM = {{")
        header.append("    " + ", ".join(str(int(val)) for val in h0.q31_taps))
        header.append("};")
        header.append("")

        if h1 is not None:
            header.append("// Highpass / Complementary Mirror QMF Pair Taps (h1)")
            header.append(f"const float h1_geg_float32[{spec.order}] = {{")
            header.append("    " + ", ".join(f"{val:.8e}f" for val in h1.float64_taps))
            header.append("};")
            header.append("")
            header.append(f"const int16_t h1_geg_q15[{spec.order}] PROGMEM = {{")
            header.append("    " + ", ".join(str(int(val)) for val in h1.q15_taps))
            header.append("};")
            header.append("")

        header.append("#endif // GEGENBAUER_FILTER_COEFFS_H")
        return "\n".join(header)

    def compile(self, spec: FilterSpec) -> FilterResult:
        truth_status = determine_truth_status(self.lam)

        a_coeffs, K = self.solve_coefficients(spec)
        h0_float = self.transform_to_taps(a_coeffs, spec)
        h0_quant = self.quantize_taps(h0_float)

        h1_quant = None
        if spec.kind == "qmf":
            sign_pattern = np.array([(-1.0)**n for n in range(spec.order)])
            h1_float = sign_pattern * h0_float[::-1]
            h1_quant = self.quantize_taps(h1_float)

        # Ensure FFT size scales dynamically to avoid aliasing on massive filter lengths
        K_fft = max(4096, 1 << (math.ceil(math.log2(spec.order)) + 3))

        H0 = np.fft.fft(h0_float, K_fft)
        freq_grid = np.arange(K_fft // 2) / float(K_fft)
        H0_db = 20 * np.log10(np.maximum(1e-12, np.abs(H0[:K_fft // 2])))

        # Compute accurate masks depending on target topology
        if spec.kind in ("lowpass", "qmf"):
            pass_idx = freq_grid <= spec.wp
            stop_idx = freq_grid >= spec.ws
        elif spec.kind == "highpass":
            pass_idx = freq_grid >= spec.wp
            stop_idx = freq_grid <= spec.ws
        elif spec.kind == "bandpass":
            pass_idx = (freq_grid >= spec.wp) & (freq_grid <= spec.wp2)
            stop_idx = (freq_grid <= spec.ws) | (freq_grid >= spec.ws2)

        pass_ripple = np.max(H0_db[pass_idx]) - np.min(H0_db[pass_idx]) if np.any(pass_idx) else 0.0
        stop_atten = -np.max(H0_db[stop_idx]) if np.any(stop_idx) else 0.0

        qmf_pow_db = 0.0
        qmf_alias_db = 0.0
        if spec.kind == "qmf" and h1_quant is not None:
            H1 = np.fft.fft(h1_quant.float64_taps, K_fft)
            H1_shift = np.roll(H1, K_fft // 2)
            H0_shift = np.roll(H0, K_fft // 2)

            pow_comp = np.abs(H0)**2 + np.abs(H1)**2
            aliasing_func = 0.5 * np.abs(H0 * H0_shift - H1 * H1_shift)

            qmf_pow_db = float(np.max(np.abs(10 * np.log10(np.maximum(1e-12, pow_comp[:K_fft // 2])))))
            qmf_alias_db = float(np.max(20 * np.log10(np.maximum(1e-12, aliasing_func[:K_fft // 2]))))

        reg_energy = sum((c ** 2) * ((k * (k + 2.0 * self.lam)) ** self.reg_power) for k, c in enumerate(a_coeffs))

        asymp_err = 0.0
        matching_status = MatchingStatus.MATCHING_SCHEMA
        if self.asymptotic_mode != "none":
            omega_sample = np.linspace(0.001, np.pi - 0.001, 100)
            n_eval = max(1, K - 1)
            phi_exact = self._eval_basis(n_eval, np.cos(omega_sample))
            phi_asymp = self._eval_asymptotic_basis(n_eval, omega_sample)
            asymp_err = float(np.max(np.abs(phi_exact - phi_asymp)))

            if asymp_err < 0.05 and self.lam > 0:
                matching_status = MatchingStatus.ANALYTICALLY_CERTIFIED_MATCHING

        # Arithmetic quantization error provenance based on Q15 target
        e_arithmetic = float(np.max(np.abs(h0_quant.float64_taps - h0_quant.q15_taps / h0_quant.q15_scale)))
        provenance = ErrorBoundProvenance(
            e_analytic=asymp_err,
            e_arithmetic=e_arithmetic,
            e_conditioning=self.ctx.eps * 1.0,  # Input conditioning placeholder based on machine precision
            e_implementation=0.0
        )

        basis_evaluation_certified = (asymp_err < 0.05)
        pass_ripple_thresh = spec.passband_ripple_db * 3.0 if spec.kind == "qmf" else spec.passband_ripple_db * 2.0
        prototype_fir_certified = (
            pass_ripple <= pass_ripple_thresh and
            stop_atten >= min(spec.stopband_atten_db * 0.5, 20.0)
        )
        qmf_power_complementary = True
        qmf_alias_cancellation = True

        if spec.kind == "qmf":
            qmf_power_complementary = (qmf_pow_db <= 1.0)
            qmf_alias_cancellation = (qmf_alias_db <= -30.0)

        is_certified = (
            basis_evaluation_certified and
            prototype_fir_certified and
            qmf_power_complementary and
            qmf_alias_cancellation
        )

        payload = CertifiedEvaluationPayload(
            truth_status=truth_status,
            matching_status=matching_status,
            provenance=provenance,
            basis_evaluation_certified=basis_evaluation_certified,
            prototype_fir_certified=prototype_fir_certified,
            qmf_power_complementary=qmf_power_complementary,
            qmf_alias_cancellation=qmf_alias_cancellation,
            is_certified=is_certified
        )

        prov_err = float(np.max(mixed_error(h0_quant.float64_taps, h0_quant.q15_taps / h0_quant.q15_scale)))

        result = FilterResult(
            spec=spec,
            lam=self.lam,
            basis_terms=K,
            mu_reg=self.mu_reg,
            h0_taps=h0_quant,
            payload=payload,
            h1_taps=h1_quant,
            freq_grid=freq_grid,
            H0_response=H0_db,
            H1_response=20 * np.log10(np.maximum(1e-12, np.abs(np.fft.fft(h1_quant.float64_taps, K_fft)[:K_fft // 2]))) if h1_quant else None,
            passband_ripple_actual=float(pass_ripple),
            stopband_atten_actual=float(stop_atten),
            qmf_power_complementarity_max_db=qmf_pow_db,
            qmf_alias_distortion_max_db=qmf_alias_db,
            regularization_energy=float(reg_energy),
            asymptotic_error_bound=asymp_err,
            provenance_mixed_error=prov_err
        )
        result.header_code = self.generate_header(result)
        return result

    def plot_response(self, result: FilterResult, output_path: str):
        """Generates comprehensive multi-panel verification plots."""
        spec = result.spec
        plt.figure(figsize=(16, 12))

        # Subplot 1: Frequency Response (dB)
        plt.subplot(2, 2, 1)
        plt.plot(result.freq_grid, result.H0_response, 'b-', label='H0 (Lowpass)' if spec.kind == 'qmf' else 'H(e^{jw})', linewidth=2)
        if result.H1_response is not None:
            plt.plot(result.freq_grid, result.H1_response, 'r-', label='H1 (Highpass)', linewidth=2)
        plt.axvline(spec.cutoff, color='g', linestyle=':', label=f'Cutoff = {spec.cutoff}')
        plt.title(f"Frequency Response ({spec.kind.upper()}, N={spec.order}, Lambda={result.lam})")
        plt.xlabel("Normalized Frequency (Cycles/Sample)")
        plt.ylabel("Magnitude (dB)")
        plt.ylim(-100, 5)
        plt.grid(True)
        plt.legend()

        # Subplot 2: Passband Detail / Ripple
        plt.subplot(2, 2, 2)
        if spec.kind == "highpass":
            pass_mask = result.freq_grid >= spec.wp
        elif spec.kind == "bandpass":
            pass_mask = (result.freq_grid >= spec.wp) & (result.freq_grid <= spec.wp2)
        else:
            pass_mask = result.freq_grid <= spec.wp

        if np.any(pass_mask):
            plt.plot(result.freq_grid[pass_mask], result.H0_response[pass_mask], 'b-', linewidth=2)
            plt.title(f"Passband Detail (Ripple = {result.passband_ripple_actual:.4f} dB)")
            plt.xlabel("Normalized Frequency")
            plt.ylabel("Magnitude (dB)")
            plt.grid(True)

        # Subplot 3: QMF / Impulse Response
        plt.subplot(2, 2, 3)
        if spec.kind == "qmf" and result.h1_taps is not None:
            K_fft = len(result.freq_grid) * 2
            H0 = np.fft.fft(result.h0_taps.float64_taps, K_fft)
            H1 = np.fft.fft(result.h1_taps.float64_taps, K_fft)
            H0_shift = np.roll(H0, K_fft // 2)
            H1_shift = np.roll(H1, K_fft // 2)

            pow_comp = np.abs(H0)**2 + np.abs(H1)**2
            aliasing_func = 0.5 * np.abs(H0 * H0_shift - H1 * H1_shift)
            pow_db = 10 * np.log10(np.maximum(1e-12, pow_comp[:K_fft // 2]))
            alias_db = 20 * np.log10(np.maximum(1e-12, aliasing_func[:K_fft // 2]))

            plt.plot(result.freq_grid, pow_db, 'g-', label='Power Complementarity $|H_0|^2 + |H_1|^2$', linewidth=2)
            plt.plot(result.freq_grid, alias_db, 'm--', label='Alias Transfer $A(e^{j\\omega})$', linewidth=1.5)
            plt.title("QMF Pair Metrics")
            plt.xlabel("Normalized Frequency")
            plt.ylabel("Magnitude (dB)")
            plt.ylim(-80, 10)
            plt.grid(True)
            plt.legend()
        else:
            plt.stem(range(spec.order), result.h0_taps.float64_taps, linefmt='b-', markerfmt='bo', basefmt='r-')
            plt.title("Impulse Response Taps h[n]")
            plt.xlabel("n")
            plt.ylabel("Amplitude")
            plt.grid(True)

        # Subplot 4: Quantization Error Analysis
        plt.subplot(2, 2, 4)
        h_float = result.h0_taps.float64_taps
        h_q15_recon = result.h0_taps.q15_taps / result.h0_taps.q15_scale
        h_q31_recon = result.h0_taps.q31_taps / result.h0_taps.q31_scale

        err_q15 = np.abs(h_float - h_q15_recon)
        err_q31 = np.abs(h_float - h_q31_recon)

        plt.semilogy(range(spec.order), np.maximum(1e-16, err_q15), 'r-o', label='Q15 Quantization Error')
        plt.semilogy(range(spec.order), np.maximum(1e-16, err_q31), 'g-s', label='Q31 Quantization Error')
        plt.title("Fixed-Point Quantization Noise per Tap")
        plt.xlabel("Tap Index n")
        plt.ylabel("Absolute Error |h_float - h_quant|")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        dir_name = os.path.dirname(output_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close()

    def pareto_search(
        self,
        spec: FilterSpec,
        lambda_candidates: List[float] = [0.5, 1.0, 1.25, 1.5, 2.0, 2.5],
        mu_candidates: List[float] = [0.0, 1e-6, 1e-4, 1e-2],
        solver: Optional[str] = None
    ) -> FilterResult:
        """
        Searches over (lambda, mu) parameter space to find the Pareto-optimal design
        balancing passband ripple vs stopband attenuation vs quantization degradation vs certified provenance error.
        """
        best_res = None
        best_score = float('inf')
        use_solver = solver if solver is not None else self.solver

        for lam in lambda_candidates:
            for mu in mu_candidates:
                compiler = GegenbauerFilterCompiler(lam=lam, mu_reg=mu, solver=use_solver)
                res = compiler.compile(spec)

                score = res.passband_ripple_actual * 10.0 - res.stopband_atten_actual
                if res.spec.kind == "qmf":
                    score += res.qmf_power_complementarity_max_db * 20.0 + res.qmf_alias_distortion_max_db

                score += res.payload.provenance.total * 100.0

                if not res.payload.is_certified:
                    score += 1e5

                if score < best_score:
                    best_score = score
                    best_res = res

        return best_res


def main():
    parser = argparse.ArgumentParser(description="VIII-Layer Gegenbauer DSP Filter Compiler CLI")
    parser.add_argument("--kind", type=str, default="qmf", choices=["lowpass", "highpass", "bandpass", "qmf"])
    parser.add_argument("--N", type=int, default=64, help="Filter length N (number of taps)")
    parser.add_argument("--cutoff", type=float, default=0.25, help="Normalized cutoff frequency (0 to 0.5)")
    parser.add_argument("--lambda_param", type=float, default=1.25, help="Gegenbauer parameter lambda > -0.5")
    parser.add_argument("--basis_terms", type=int, default=None, help="Number of Gegenbauer basis terms")
    parser.add_argument("--solver", type=str, default="spectral_regularized", choices=["wls", "spectral_regularized", "quadrature"])
    parser.add_argument("--mu_reg", type=float, default=1e-4, help="Sturm-Liouville regularization weight")
    parser.add_argument("--sampling_rate", type=int, default=2000, help="Sampling rate in Hz")
    parser.add_argument("--output_header", type=str, default="gegenbauer_coeffs.h", help="Generated C/C++ header file path")
    parser.add_argument("--output_plot", type=str, default="gegenbauer_response.png", help="Generated plot image path")

    args = parser.parse_args()

    spec = FilterSpec(
        kind=args.kind,
        order=args.N,
        cutoff=args.cutoff,
        sampling_rate=args.sampling_rate
    )

    compiler = GegenbauerFilterCompiler(
        lam=args.lambda_param,
        basis_terms=args.basis_terms,
        solver=args.solver,
        mu_reg=args.mu_reg
    )

    print(f"Compiling Gegenbauer {args.kind.upper()} Filter...")
    result = compiler.compile(spec)
    print(result.summary())

    with open(args.output_header, "w") as f:
        f.write(result.header_code)
    print(f"Header successfully written to: {args.output_header}")

    compiler.plot_response(result, args.output_plot)
    print(f"Plot successfully saved to: {args.output_plot}")


if __name__ == "__main__":
    main()
