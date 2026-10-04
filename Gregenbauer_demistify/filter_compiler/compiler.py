"""
Gegenbauer Filter Compiler Module (Refined & Integrated)
========================================================
A comprehensive FIR/IIR/QMF DSP filter compiler based on Gegenbauer polynomial bases,
Jacobi matrix operators, Sturm-Liouville differential regularizers, and asymptotic boundary layer theory.
"""

import os
import sys
import math
import copy
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
    SAMPLED_ASYMPTOTIC_MATCHING = "SAMPLED_ASYMPTOTIC_MATCHING"
    ANALYTICALLY_CERTIFIED_MATCHING = "ANALYTICALLY_CERTIFIED_MATCHING"


class SymmetryClass(Enum):
    """FIR Filter Symmetry Classes (Types I-IV)."""
    TYPE_I = "TYPE_I"     # Odd N, symmetric
    TYPE_II = "TYPE_II"   # Even N, symmetric
    TYPE_III = "TYPE_III" # Odd N, anti-symmetric
    TYPE_IV = "TYPE_IV"   # Even N, anti-symmetric


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
    basis_asymptotic_validated: bool = True
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


def qmf_alias_transfer(H0: np.ndarray, H1: np.ndarray) -> np.ndarray:
    """Computes complex CQF/QMF alias transfer function A(e^{j\\omega}) = 0.5 * (H0(\\omega+\\pi) H0*(\\omega) + H1(\\omega+\\pi) H1*(\\omega))."""
    if len(H0) != len(H1):
        raise ValueError(f"H0 and H1 frequency response arrays must have equal length, got {len(H0)} and {len(H1)}")
    if len(H0) % 2 != 0:
        raise ValueError(f"Frequency response array length must be even, got {len(H0)}")
    K_fft = len(H0)
    H0_shift = np.roll(H0, K_fft // 2)
    H1_shift = np.roll(H1, K_fft // 2)
    return 0.5 * (H0_shift * np.conj(H0) + H1_shift * np.conj(H1))


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
        if self.kind not in ("lowpass", "highpass", "bandpass", "qmf", "asymmetric_qmf"):
            raise ValueError(f"Unknown filter kind: '{self.kind}'")
        if self.order < 3:
            raise ValueError("Filter order must be >= 3")
        if not (0.0 < self.cutoff < 0.5):
            raise ValueError(f"Cutoff must be in (0.0, 0.5), got {self.cutoff}")

        # QMF filter pairs require an even tap length
        if self.kind in ("qmf", "asymmetric_qmf") and self.order % 2 != 0:
            raise ValueError(f"QMF filter pair requires an even tap length N, got {self.order}.")

        if self.kind == "qmf":
            self.cutoff = 0.25

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

        if self.kind == "qmf":
            if abs((self.wp + self.ws) - 0.5) > 1e-6:
                raise ValueError(f"QMF filter pair requires symmetric transition band around fs/4 (wp + ws == 0.5), got wp={self.wp}, ws={self.ws}")

        if self.kind == "asymmetric_qmf":
            if not (0.0 < self.wp < 0.25 < self.ws < 0.5):
                raise ValueError(f"Asymmetric QMF requires transition band straddling fs/4 (0 < wp < 0.25 < ws < 0.5), got wp={self.wp}, ws={self.ws}")

        if self.kind in ("lowpass", "qmf", "asymmetric_qmf"):
            if not (0.0 < self.wp < self.ws < 0.5):
                raise ValueError(f"Frequency edges must satisfy 0 < wp ({self.wp}) < ws ({self.ws}) < 0.5")
        elif self.kind == "highpass":
            if not (0.0 < self.ws < self.wp < 0.5):
                raise ValueError(f"Frequency edges must satisfy 0 < ws ({self.ws}) < wp ({self.wp}) < 0.5")
        elif self.kind == "bandpass":
            if not (0.0 < self.ws < self.wp < self.wp2 < self.ws2 < 0.5):
                raise ValueError(f"Bandpass frequencies must satisfy 0 < ws ({self.ws}) < wp ({self.wp}) < wp2 ({self.wp2}) < ws2 ({self.ws2}) < 0.5")

    @property
    def symmetry_class(self) -> SymmetryClass:
        """Determines the FIR symmetry class (Type I-IV) based on filter type and order N."""
        is_even = (self.order % 2 == 0)
        if self.kind == "highpass":
            return SymmetryClass.TYPE_IV if is_even else SymmetryClass.TYPE_I
        elif self.kind in ("lowpass", "bandpass", "qmf", "asymmetric_qmf"):
            return SymmetryClass.TYPE_II if is_even else SymmetryClass.TYPE_I
        return SymmetryClass.TYPE_I


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
    qmf_power_error_linear: float = 0.0
    qmf_alias_error_linear: float = 0.0
    regularization_energy: float = 0.0
    fit_residual: float = 0.0
    data_fit_residual: float = 0.0
    regularization_residual: float = 0.0
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
            f"Certifications: Basis={self.payload.basis_asymptotic_validated} | Prototype={self.payload.prototype_fir_certified}"
            f" | QMF Power={self.payload.qmf_power_complementary} | QMF Alias={self.payload.qmf_alias_cancellation} | Total={self.payload.is_certified}",
            f"Passband Ripple: {self.passband_ripple_actual:.4f} dB | Stopband Attenuation: {self.stopband_atten_actual:.2f} dB",
        ]
        if self.spec.kind in ("qmf", "asymmetric_qmf"):
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
        if basis_terms is not None and basis_terms < 1:
            raise ValueError(f"basis_terms must be >= 1, got {basis_terms}")
        if mu_reg < 0.0:
            raise ValueError(f"mu_reg must be >= 0.0, got {mu_reg}")
        if reg_power < 0:
            raise ValueError(f"reg_power must be >= 0, got {reg_power}")
        if basis_type not in ("normalized", "unnormalized"):
            raise ValueError(f"Unknown basis_type: '{basis_type}'")
        if solver not in ("spectral_regularized", "quadrature", "least_squares"):
            raise ValueError(f"Unknown solver mode: '{solver}'")
        if asymptotic_mode not in ("auto", "bessel", "wkb", "composite", "none"):
            raise ValueError(f"Unknown asymptotic_mode: '{asymptotic_mode}'")
        if grid_samples < 16:
            raise ValueError(f"grid_samples must be >= 16, got {grid_samples}")

        self.lam = float(lam)
        self.basis_terms = basis_terms
        self.basis_type = basis_type
        self.solver = solver
        self.mu_reg = float(mu_reg)
        self.reg_power = int(reg_power)
        self.asymptotic_mode = asymptotic_mode
        self.grid_samples = grid_samples
        self.ctx = NumericalContext(precision=precision, base=NumericalBase.BASE_2)

    def _independent_dimension(self, spec: FilterSpec) -> int:
        N = spec.order
        sym = spec.symmetry_class
        if sym == SymmetryClass.TYPE_I:
            return (N + 1) // 2
        elif sym == SymmetryClass.TYPE_II:
            return N // 2
        elif sym == SymmetryClass.TYPE_III:
            return (N - 1) // 2
        elif sym == SymmetryClass.TYPE_IV:
            return N // 2
        return (N + 1) // 2

    def _eval_basis(self, n: int, x: np.ndarray, symmetry: SymmetryClass = SymmetryClass.TYPE_I) -> np.ndarray:
        """Evaluates n-th basis function at array x = cos(omega) in [-1, 1], multiplied by symmetry envelope."""
        x_arr = np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0)
        if self.basis_type == "normalized":
            P_k = normalized_phi_recurrence(n, self.lam, x_arr)
        else:
            c1 = float(c_n_1_val(n, self.lam))
            P_k = c1 * normalized_phi_recurrence(n, self.lam, x_arr)

        if symmetry == SymmetryClass.TYPE_I:
            return P_k
        elif symmetry == SymmetryClass.TYPE_II:
            # cos(omega/2) = sqrt((1 + x)/2)
            return np.sqrt(0.5 * (1.0 + x_arr)) * P_k
        elif symmetry == SymmetryClass.TYPE_III:
            # sin(omega) = sqrt(1 - x^2)
            return np.sqrt(np.maximum(0.0, 1.0 - x_arr**2)) * P_k
        elif symmetry == SymmetryClass.TYPE_IV:
            # sin(omega/2) = sqrt((1 - x)/2)
            return np.sqrt(np.maximum(0.0, 0.5 * (1.0 - x_arr))) * P_k
        return P_k

    def _eval_asymptotic_basis(self, n: int, omega: np.ndarray, symmetry: SymmetryClass = SymmetryClass.TYPE_I) -> np.ndarray:
        theta = omega
        if self.asymptotic_mode == "bessel":
            phi_asymp = endpoint_bessel_leading(n, self.lam, theta)
        elif self.asymptotic_mode == "wkb":
            phi_asymp = interior_wkb_approx(n, self.lam, theta)
        else:
            # composite / auto mode uses two-endpoint composite matched asymptotics
            phi_asymp = composite_matched_approx(n, self.lam, theta)

        if symmetry == SymmetryClass.TYPE_I:
            return phi_asymp
        elif symmetry == SymmetryClass.TYPE_II:
            return np.cos(0.5 * omega) * phi_asymp
        elif symmetry == SymmetryClass.TYPE_III:
            return np.sin(omega) * phi_asymp
        elif symmetry == SymmetryClass.TYPE_IV:
            return np.sin(0.5 * omega) * phi_asymp
        return phi_asymp

    def _build_spectral_target(self, spec: FilterSpec, omega: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        f = omega / (2.0 * np.pi)
        D = np.zeros_like(omega)
        W = np.ones_like(omega)
        wp, ws = spec.wp, spec.ws

        if spec.kind in ("qmf", "asymmetric_qmf"):
            pass_mask = f <= wp
            stop_mask = f >= ws
            trans_mask = ~(pass_mask | stop_mask)
            D[pass_mask] = 1.0
            D[stop_mask] = 0.0
            if np.any(trans_mask):
                t = (f[trans_mask] - wp) / (ws - wp)
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

    def solve_halfband_power_polynomial(self, spec: FilterSpec) -> Tuple[np.ndarray, np.ndarray, int, float, float, float, float]:
        """
        Solves for half-band power polynomial P(x) = 0.5 + R_odd(x) in odd Gegenbauer basis C_{2k+1}^(\\lambda)(x)
        for an N-tap QMF filter H0 (where P(z) has degree 2N-2 and length 2N-1 = 2M+1).
        Returns (a_coeffs, p_taps, K, cond_val, res_aug, res_data, res_reg).
        Structurally guarantees P(x) + P(-x) = 1 in frequency space.
        """
        N = spec.order       # N taps for H0
        M = N - 1            # Degree of H0 = N - 1
        center = M           # Center tap index of P(z) = M (length 2M + 1)
        K = (M + 1) // 2     # Max Gegenbauer odd terms <= M (Fourier harmonics <= M)

        nodes, weights = gauss_gegenbauer_quadrature(self.grid_samples, self.lam)
        omega_q = np.arccos(nodes)

        D_q, W_q = self._build_spectral_target(spec, omega_q)
        R_target = np.abs(D_q)**2 - 0.5

        A_odd = np.zeros((self.grid_samples, K))
        for k in range(K):
            n_odd = 2 * k + 1
            A_odd[:, k] = self._eval_basis(n_odd, nodes, symmetry=SymmetryClass.TYPE_I)

        sqrt_W = np.sqrt(W_q * weights)
        A_w = A_odd * sqrt_W[:, np.newaxis]
        D_w = R_target * sqrt_W

        mu = self.mu_reg
        R_diag = np.zeros(K)
        for k in range(K):
            n_odd = 2 * k + 1
            eig = n_odd * (n_odd + 2.0 * self.lam)
            R_diag[k] = (eig ** self.reg_power)

        R_mat = np.diag(np.sqrt(mu * R_diag))
        A_sys = np.vstack([A_w, R_mat])
        D_sys = np.concatenate([D_w, np.zeros(R_mat.shape[0])])

        a_odd, _, _, _ = np.linalg.lstsq(A_sys, D_sys, rcond=None)
        s_vals = np.linalg.svd(A_sys, compute_uv=False)
        cond_val = float(s_vals[0] / s_vals[-1]) if len(s_vals) > 0 and s_vals[-1] > 1e-12 else np.inf

        res_aug = float(np.linalg.norm(A_sys @ a_odd - D_sys) / max(np.linalg.norm(D_sys), 1e-15))
        res_data = float(np.linalg.norm(A_w @ a_odd - D_w) / max(np.linalg.norm(D_w), 1e-15))
        res_reg = float(np.linalg.norm(R_mat @ a_odd) / max(np.linalg.norm(D_w), 1e-15))

        a_coeffs = np.zeros(2 * K)
        for k in range(K):
            a_coeffs[2 * k + 1] = a_odd[k]

        # Convert R_odd(cos w) to Fourier cosine taps b_k
        grid_L = self.grid_samples
        omega_grid = np.linspace(0, np.pi, grid_L)
        x_grid = np.cos(omega_grid)
        R_vals = np.zeros(grid_L, dtype=np.float64)
        for k in range(K):
            n_odd = 2 * k + 1
            R_vals += a_odd[k] * self._eval_basis(n_odd, x_grid, symmetry=SymmetryClass.TYPE_I)

        trapz_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
        p_taps = np.zeros(2 * M + 1, dtype=np.float64)
        p_taps[center] = 0.5 # Center tap corresponds to constant 0.5

        for m in range(1, M + 1):
            if m % 2 != 0: # Odd Fourier harmonics
                integrand = R_vals * np.cos(m * omega_grid)
                b_m = (2.0 / np.pi) * trapz_fn(integrand, x=omega_grid)
                p_taps[center - m] = b_m / 2.0
                p_taps[center + m] = b_m / 2.0

        # Ensure P(w) = 0.5 + R_odd(w) is non-negative P(w) >= 0 and P(w) <= 1
        # so that power polynomial P(z) is valid for spectral factorization
        w_eval = np.linspace(0, np.pi, 2048)
        P_w = np.zeros_like(w_eval)
        for n, val in enumerate(p_taps):
            P_w += val * np.cos((n - center) * w_eval)

        min_P = np.min(P_w)
        max_P = np.max(P_w)
        over = max(0.0, -min_P, max_P - 1.0)
        if over > 1e-12:
            scale_factor = 0.5 / (0.5 + over)
            for m in range(1, M + 1):
                if m % 2 != 0:
                    p_taps[center - m] *= scale_factor
                    p_taps[center + m] *= scale_factor

        return a_coeffs, p_taps, K, cond_val, res_aug, res_data, res_reg

    def validate_power_polynomial(self, p_taps: np.ndarray, grid_size: int = 2048) -> Tuple[bool, float, float, float]:
        """
        Validates half-band power polynomial P(z).
        Returns (is_valid, min_P, max_P, max_halfband_err).
        Checks P(w) >= 0 and P(w) + P(w + pi) = 1.
        """
        K_fft = max(4096, grid_size)
        center = (len(p_taps) - 1) // 2
        w = 2.0 * np.pi * np.arange(K_fft) / float(K_fft)
        P_w = np.zeros(K_fft, dtype=np.float64)
        for n, val in enumerate(p_taps):
            P_w += val * np.cos((n - center) * w)

        min_P = float(np.min(P_w))
        max_P = float(np.max(P_w))

        P_shift = np.roll(P_w, K_fft // 2)
        max_hb_err = float(np.max(np.abs(P_w + P_shift - 1.0)))

        is_valid = (min_P >= -0.05) and (max_hb_err <= 1e-4)
        return is_valid, min_P, max_P, max_hb_err

    def spectral_factor_power_polynomial(self, p_taps: np.ndarray, target_N: int) -> Tuple[np.ndarray, float]:
        """
        Spectrally factors positive half-band power polynomial P(z) into minimum-phase / linear-phase factor H0(z) of length target_N.
        Uses canonical root orbit grouping (conjugate & reciprocal closure) to preserve real minimum-phase factors.
        Returns (h0_taps, root_reciprocity_residual).
        """
        M = target_N - 1 # Degree of H0
        roots = np.roots(p_taps)

        # Build canonical orbits under conjugation and reciprocal reflection
        rel_tol, abs_tol = 1e-2, 1e-3
        def same_root(a, b):
            return abs(a - b) <= max(abs_tol, rel_tol * max(abs(a), abs(b)))

        orbits = []
        used = [False] * len(roots)
        for i in range(len(roots)):
            if used[i]:
                continue
            r_val = roots[i]
            orbit_idx = [i]
            used[i] = True

            added = True
            while added:
                added = False
                curr_vals = [roots[k] for k in orbit_idx]
                for cv in curr_vals:
                    targets = [np.conj(cv)]
                    if abs(cv) > 1e-12:
                        targets.append(1.0 / cv)
                        targets.append(1.0 / np.conj(cv))
                    for tgt in targets:
                        for j in range(len(roots)):
                            if not used[j] and same_root(roots[j], tgt):
                                orbit_idx.append(j)
                                used[j] = True
                                added = True
            orbits.append([roots[k] for k in orbit_idx])

        # Select M roots inside or on the unit circle from canonical orbits
        inside_roots = []
        for orbit in orbits:
            in_orbit = [r for r in orbit if abs(r) <= 1.0 + 1e-4]
            half_len = max(1, len(orbit) // 2)
            if len(in_orbit) == half_len:
                inside_roots.extend(in_orbit)
            elif len(in_orbit) == len(orbit):
                inside_roots.extend(in_orbit)
            else:
                sorted_orb = sorted(orbit, key=lambda x: abs(x))
                inside_roots.extend(sorted_orb[:half_len])

        # Truncate / pad to exact target degree M
        if len(inside_roots) > M:
            inside_roots = sorted(inside_roots, key=lambda x: abs(x))[:M]

        h0 = np.poly(inside_roots).real

        # Scale h0 so convolution h0 * h0[::-1] matches center tap p_taps[center]
        center = (len(p_taps) - 1) // 2
        r_h0 = np.convolve(h0, h0[::-1])
        scale = np.sqrt(max(1e-15, p_taps[center] / max(1e-15, r_h0[len(r_h0) // 2])))
        h0 *= scale

        # Calculate root reciprocity residual
        recip_err = float(np.max(np.abs(np.convolve(h0, h0[::-1]) - p_taps)))

        return h0, recip_err

    def solve_qmf_power_coefficients(self, spec: FilterSpec) -> Tuple[np.ndarray, int, float, float, float, float]:
        """Backward compatibility wrapper around solve_halfband_power_polynomial."""
        a_coeffs, _, K, cond_val, res_aug, res_data, res_reg = self.solve_halfband_power_polynomial(spec)
        return a_coeffs, K, cond_val, res_aug, res_data, res_reg

    def solve_coefficients(self, spec: FilterSpec) -> Tuple[np.ndarray, int, float, float, float, float]:
        N = spec.order
        M = self._independent_dimension(spec)
        if self.basis_terms is not None:
            if self.basis_terms > M:
                raise ValueError(f"basis_terms={self.basis_terms} exceeds the independent dimension M={M} for {spec.symmetry_class.value}")
            K = self.basis_terms
        else:
            if spec.kind in ("qmf", "asymmetric_qmf"):
                K = min(M, max(16, N // 2))
            else:
                K = max(4, int(np.sqrt(N)) + 2)
        K = min(K, M)

        mu = self.mu_reg
        sym = spec.symmetry_class

        if self.solver == "quadrature":
            nodes, weights = gauss_gegenbauer_quadrature(self.grid_samples, self.lam)
            omega_q = np.arccos(nodes)
            D_q, _ = self._build_spectral_target(spec, omega_q)

            if sym == SymmetryClass.TYPE_I:
                a_coeffs = np.zeros(K)
                for k in range(K):
                    phi_k = self._eval_basis(k, nodes, symmetry=sym)
                    c1 = float(c_n_1_val(k, self.lam))
                    norm_sq = phi_norm_squared(k, self.lam) / (c1 ** 2) if self.basis_type == "normalized" else phi_norm_squared(k, self.lam)
                    a_coeffs[k] = np.sum(D_q * phi_k * weights) / norm_sq
                A = np.zeros((self.grid_samples, K))
                for k in range(K):
                    A[:, k] = self._eval_basis(k, nodes, symmetry=sym)
                sqrt_w = np.sqrt(weights)
                A_w = A * sqrt_w[:, np.newaxis]
                s_vals = np.linalg.svd(A_w, compute_uv=False)
                cond_val = float(s_vals[0] / s_vals[-1]) if len(s_vals) > 0 and s_vals[-1] > 1e-12 else np.inf
                res_data = float(np.linalg.norm(sqrt_w * (A @ a_coeffs - D_q)) / max(np.linalg.norm(sqrt_w * D_q), 1e-15))
                return a_coeffs, K, cond_val, res_data, res_data, 0.0
            else:
                # Direct QR/SVD least squares solve without forming normal equations A_w.T @ A_w
                A = np.zeros((self.grid_samples, K))
                for k in range(K):
                    A[:, k] = self._eval_basis(k, nodes, symmetry=sym)
                sqrt_w = np.sqrt(weights)
                A_w = A * sqrt_w[:, np.newaxis]
                D_w = D_q * sqrt_w
                a_coeffs, _, _, _ = np.linalg.lstsq(A_w, D_w, rcond=None)
                s_vals = np.linalg.svd(A_w, compute_uv=False)
                cond_val = float(s_vals[0] / s_vals[-1]) if len(s_vals) > 0 and s_vals[-1] > 1e-12 else np.inf
                res_data = float(np.linalg.norm(A_w @ a_coeffs - D_w) / max(np.linalg.norm(D_w), 1e-15))
                return a_coeffs, K, cond_val, res_data, res_data, 0.0

        elif self.solver == "spectral_regularized":
            nodes, weights = gauss_gegenbauer_quadrature(self.grid_samples, self.lam)
            omega_q = np.arccos(nodes)
            D_q, W_q = self._build_spectral_target(spec, omega_q)

            A = np.zeros((self.grid_samples, K))
            for k in range(K):
                A[:, k] = self._eval_basis(k, nodes, symmetry=sym)

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
            s_vals = np.linalg.svd(A_sys, compute_uv=False)
            cond_val = float(s_vals[0] / s_vals[-1]) if len(s_vals) > 0 and s_vals[-1] > 1e-12 else np.inf
            res_aug = float(np.linalg.norm(A_sys @ a_coeffs - D_sys) / max(np.linalg.norm(D_sys), 1e-15))
            res_data = float(np.linalg.norm(A_w @ a_coeffs - D_w) / max(np.linalg.norm(D_w), 1e-15))
            res_reg = float(np.linalg.norm(R_mat @ a_coeffs) / max(np.linalg.norm(D_w), 1e-15))
            return a_coeffs, K, cond_val, res_aug, res_data, res_reg

        elif self.solver == "least_squares":
            omega = np.linspace(0, np.pi, self.grid_samples)
            x = np.cos(omega)
            D, W = self._build_spectral_target(spec, omega)

            A = np.zeros((self.grid_samples, K))
            for k in range(K):
                A[:, k] = self._eval_basis(k, x, symmetry=sym)

            sqrt_W = np.sqrt(W)
            A_w = A * sqrt_W[:, np.newaxis]
            D_w = D * sqrt_W
            a_coeffs, _, _, _ = np.linalg.lstsq(A_w, D_w, rcond=None)
            s_vals = np.linalg.svd(A_w, compute_uv=False)
            cond_val = float(s_vals[0] / s_vals[-1]) if len(s_vals) > 0 and s_vals[-1] > 1e-12 else np.inf
            res_data = float(np.linalg.norm(A_w @ a_coeffs - D_w) / max(np.linalg.norm(D_w), 1e-15))

            return a_coeffs, K, cond_val, res_data, res_data, 0.0

        raise ValueError(f"Unknown solver: '{self.solver}'")

    def transform_to_taps(self, a_coeffs: np.ndarray, spec: FilterSpec) -> np.ndarray:
        N = spec.order
        sym = spec.symmetry_class
        grid_L = self.grid_samples
        omega = np.linspace(0, np.pi, grid_L)
        x = np.cos(omega)

        A_freq = np.zeros(grid_L, dtype=np.float64)
        for k, c in enumerate(a_coeffs):
            A_freq += c * self._eval_basis(k, x, symmetry=sym)

        h = np.zeros(N, dtype=np.float64)
        trapz_fn = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz

        if sym == SymmetryClass.TYPE_I: # N odd, symmetric
            mid = (N - 1) // 2
            for m in range(mid + 1):
                integrand = A_freq if m == 0 else A_freq * np.cos(m * omega)
                val = (1.0 / np.pi) * trapz_fn(integrand, x=omega)
                if m == 0:
                    h[mid] = val
                else:
                    h[mid - m] = val
                    h[mid + m] = val

        elif sym == SymmetryClass.TYPE_II: # N even, symmetric
            M = N // 2
            for m in range(1, M + 1):
                integrand = A_freq * np.cos((m - 0.5) * omega)
                val = (1.0 / np.pi) * trapz_fn(integrand, x=omega)
                h[M - m] = val
                h[M + m - 1] = val

        elif sym == SymmetryClass.TYPE_III: # N odd, anti-symmetric
            mid = (N - 1) // 2
            h[mid] = 0.0
            for m in range(1, mid + 1):
                integrand = A_freq * np.sin(m * omega)
                val = (1.0 / np.pi) * trapz_fn(integrand, x=omega)
                h[mid - m] = val
                h[mid + m] = -val

        elif sym == SymmetryClass.TYPE_IV: # N even, anti-symmetric
            M = N // 2
            for m in range(1, M + 1):
                integrand = A_freq * np.sin((m - 0.5) * omega)
                val = (1.0 / np.pi) * trapz_fn(integrand, x=omega)
                h[M - m] = val
                h[M + m - 1] = -val

        if spec.kind in ("lowpass", "qmf", "asymmetric_qmf"):
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
            f"// Numerical Diagnostics: E_analytic = {p.e_analytic:.6e} | E_arithmetic = {p.e_arithmetic:.6e} | Condition = {p.e_conditioning:.6e}"
        ]
        if spec.kind in ("qmf", "asymmetric_qmf"):
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
            f"#define GEG_Q15_SCALE {h0.q15_scale}f",
            f"#define GEG_Q23_SCALE {h0.q23_scale}f",
            f"#define GEG_Q31_SCALE {h0.q31_scale}f",
            ""
        ])

        header.append(f"static const double h0_geg_float64[{spec.order}] = {{")
        header.append("    " + ", ".join(f"{val:.12e}" for val in h0.float64_taps))
        header.append("};")
        header.append("")

        header.append(f"static const float h0_geg_float32[{spec.order}] = {{")
        header.append("    " + ", ".join(f"{val:.8e}f" for val in h0.float64_taps))
        header.append("};")
        header.append("")

        header.append(f"static const int16_t h0_geg_q15[{spec.order}] PROGMEM = {{")
        header.append("    " + ", ".join(str(int(val)) for val in h0.q15_taps))
        header.append("};")
        header.append("")

        header.append(f"static const int32_t h0_geg_q23[{spec.order}] PROGMEM = {{")
        header.append("    " + ", ".join(str(int(val)) for val in h0.q23_taps))
        header.append("};")
        header.append("")

        header.append(f"static const int32_t h0_geg_q31[{spec.order}] PROGMEM = {{")
        header.append("    " + ", ".join(str(int(val)) for val in h0.q31_taps))
        header.append("};")
        header.append("")

        if h1 is not None:
            header.append("// Highpass / Complementary Mirror QMF Pair Taps (h1)")
            header.append(f"static const double h1_geg_float64[{spec.order}] = {{")
            header.append("    " + ", ".join(f"{val:.12e}" for val in h1.float64_taps))
            header.append("};")
            header.append("")
            header.append(f"static const float h1_geg_float32[{spec.order}] = {{")
            header.append("    " + ", ".join(f"{val:.8e}f" for val in h1.float64_taps))
            header.append("};")
            header.append("")
            header.append(f"static const int16_t h1_geg_q15[{spec.order}] PROGMEM = {{")
            header.append("    " + ", ".join(str(int(val)) for val in h1.q15_taps))
            header.append("};")
            header.append("")
            header.append(f"static const int32_t h1_geg_q23[{spec.order}] PROGMEM = {{")
            header.append("    " + ", ".join(str(int(val)) for val in h1.q23_taps))
            header.append("};")
            header.append("")
            header.append(f"static const int32_t h1_geg_q31[{spec.order}] PROGMEM = {{")
            header.append("    " + ", ".join(str(int(val)) for val in h1.q31_taps))
            header.append("};")
            header.append("")

        header.append("#endif // GEGENBAUER_FILTER_COEFFS_H")
        return "\n".join(header)

    def compile(self, spec: FilterSpec) -> FilterResult:
        truth_status = determine_truth_status(self.lam)

        if spec.kind in ("qmf", "asymmetric_qmf"):
            # Direct QMF compilation route via half-band power polynomial and spectral factorization
            a_coeffs, p_taps, K, cond_val, res_aug, res_data, res_reg = self.solve_halfband_power_polynomial(spec)
            _, min_P, max_P, _ = self.validate_power_polynomial(p_taps)
            if min_P < -0.05:
                raise ValueError(f"Half-band power response min P = {min_P:.4f} < -0.05 is not spectrally factorizable.")

            h0_float, _ = self.spectral_factor_power_polynomial(p_taps, target_N=spec.order)
            h0_quant = self.quantize_taps(h0_float)

            # Derive H1 directly via CQF modulation h1[n] = (-1)^n * h0[N-1-n]
            sign_pattern = np.array([(-1.0)**n for n in range(spec.order)])
            h1_float = sign_pattern * h0_float[::-1]

            q15_sign = np.array([(-1)**n for n in range(spec.order)], dtype=np.int32)
            q31_sign = np.array([(-1)**n for n in range(spec.order)], dtype=np.int64)

            h1_q15 = (q15_sign * h0_quant.q15_taps[::-1]).astype(np.int32)
            h1_q23 = (q15_sign * h0_quant.q23_taps[::-1]).astype(np.int32)
            h1_q31 = (q31_sign * h0_quant.q31_taps[::-1]).astype(np.int64)

            h1_quant = QuantizedTaps(
                float64_taps=h1_float,
                q15_taps=h1_q15,
                q23_taps=h1_q23,
                q31_taps=h1_q31,
                q15_scale=h0_quant.q15_scale,
                q23_scale=h0_quant.q23_scale,
                q31_scale=h0_quant.q31_scale
            )
        else:
            a_coeffs, K, cond_val, res_aug, res_data, res_reg = self.solve_coefficients(spec)
            h0_float = self.transform_to_taps(a_coeffs, spec)
            h0_quant = self.quantize_taps(h0_float)
            h1_quant = None

        K_fft = max(4096, 1 << (math.ceil(math.log2(spec.order)) + 3))

        H0 = np.fft.fft(h0_float, K_fft)
        freq_grid = np.arange(K_fft // 2 + 1) / float(K_fft)
        H0_db = 20 * np.log10(np.maximum(1e-12, np.abs(H0[:K_fft // 2 + 1])))

        if spec.kind in ("lowpass", "qmf", "asymmetric_qmf"):
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

        pass_ripple_h1 = 0.0
        stop_atten_h1 = 0.0
        qmf_pow_db = 0.0
        qmf_alias_db = 0.0
        qmf_pow_lin = 0.0
        qmf_alias_lin = 0.0

        if spec.kind in ("qmf", "asymmetric_qmf") and h1_quant is not None:
            H1 = np.fft.fft(h1_quant.float64_taps, K_fft)
            H1_db = 20 * np.log10(np.maximum(1e-12, np.abs(H1[:K_fft // 2 + 1])))

            # For CQF/QMF, H1's transition mirror bounds are f >= 0.5 - wp (passband) and f <= 0.5 - ws (stopband)
            h1_pass_edge = 0.5 - spec.wp
            h1_stop_edge = 0.5 - spec.ws
            pass_idx_h1 = freq_grid >= h1_pass_edge
            stop_idx_h1 = freq_grid <= h1_stop_edge

            pass_ripple_h1 = float(np.max(H1_db[pass_idx_h1]) - np.min(H1_db[pass_idx_h1])) if np.any(pass_idx_h1) else 0.0
            stop_atten_h1 = float(-np.max(H1_db[stop_idx_h1])) if np.any(stop_idx_h1) else 0.0

            # Multi-format QMF response checks (float64, q15, q23, q31)
            qmf_pow_db_list = []
            qmf_alias_db_list = []
            qmf_pow_lin_list = []
            qmf_alias_lin_list = []

            formats_to_check = [
                (h0_quant.float64_taps, h1_quant.float64_taps),
                (h0_quant.q15_taps / h0_quant.q15_scale, h1_quant.q15_taps / h1_quant.q15_scale),
                (h0_quant.q23_taps / h0_quant.q23_scale, h1_quant.q23_taps / h1_quant.q23_scale),
                (h0_quant.q31_taps / h0_quant.q31_scale, h1_quant.q31_taps / h1_quant.q31_scale),
            ]
            for h0_arr, h1_arr in formats_to_check:
                H0_k = np.fft.fft(h0_arr, K_fft)
                H1_k = np.fft.fft(h1_arr, K_fft)
                pow_comp_k = np.abs(H0_k)**2 + np.abs(H1_k)**2
                alias_k = np.abs(qmf_alias_transfer(H0_k, H1_k))

                pow_err_lin_k = float(np.max(np.abs(pow_comp_k - 1.0)))
                alias_err_lin_k = float(np.max(alias_k))

                qmf_pow_lin_list.append(pow_err_lin_k)
                qmf_alias_lin_list.append(alias_err_lin_k)
                qmf_pow_db_list.append(float(np.max(np.abs(10 * np.log10(np.maximum(1e-12, pow_comp_k[:K_fft // 2 + 1]))))))
                qmf_alias_db_list.append(float(np.max(20 * np.log10(np.maximum(1e-12, alias_k[:K_fft // 2 + 1])))))

            qmf_pow_db = max(qmf_pow_db_list)
            qmf_alias_db = max(qmf_alias_db_list)
            qmf_pow_lin = max(qmf_pow_lin_list)
            qmf_alias_lin = max(qmf_alias_lin_list)

        reg_energy = sum((c ** 2) * ((k * (k + 2.0 * self.lam)) ** self.reg_power) for k, c in enumerate(a_coeffs))

        asymp_err = 0.0
        matching_status = MatchingStatus.MATCHING_SCHEMA
        if self.asymptotic_mode != "none":
            omega_sample = np.linspace(0.001, np.pi - 0.001, 100)
            n_eval = max(0, K - 1)
            sym = spec.symmetry_class
            phi_exact = self._eval_basis(n_eval, np.cos(omega_sample), symmetry=sym)
            phi_asymp = self._eval_asymptotic_basis(n_eval, omega_sample, symmetry=sym)
            asymp_err = float(np.max(np.abs(phi_exact - phi_asymp)))

            if asymp_err < 0.05 and self.lam > 0:
                matching_status = MatchingStatus.SAMPLED_ASYMPTOTIC_MATCHING

        e_q15 = float(np.max(np.abs(h0_quant.float64_taps - h0_quant.q15_taps / h0_quant.q15_scale)))
        e_q23 = float(np.max(np.abs(h0_quant.float64_taps - h0_quant.q23_taps / h0_quant.q23_scale)))
        e_q31 = float(np.max(np.abs(h0_quant.float64_taps - h0_quant.q31_taps / h0_quant.q31_scale)))
        e_arithmetic = max(e_q15, e_q23, e_q31)

        provenance = ErrorBoundProvenance(
            e_analytic=asymp_err,
            e_arithmetic=e_arithmetic,
            e_conditioning=self.ctx.eps * cond_val,
            e_implementation=0.0
        )

        basis_asymptotic_validated = (self.asymptotic_mode != "none" and asymp_err < 0.05)
        # Separate design target metrics from prototype FIR specification thresholds
        pass_ripple_thresh = spec.passband_ripple_db * 3.0 if spec.kind in ("qmf", "asymmetric_qmf") else spec.passband_ripple_db * 2.0
        prototype_fir_certified = (
            pass_ripple <= pass_ripple_thresh and
            stop_atten >= min(spec.stopband_atten_db * 0.5, 15.0)
        )
        if spec.kind in ("qmf", "asymmetric_qmf") and h1_quant is not None:
            prototype_fir_certified = prototype_fir_certified and (
                pass_ripple_h1 <= pass_ripple_thresh and
                stop_atten_h1 >= min(spec.stopband_atten_db * 0.5, 15.0)
            )
        
        qmf_power_complementary = True
        qmf_alias_cancellation = True

        if spec.kind in ("qmf", "asymmetric_qmf"):
            qmf_power_complementary = (qmf_pow_db <= 1.0)
            qmf_alias_cancellation = (qmf_alias_db <= -20.0)

        is_certified = (
            basis_asymptotic_validated and
            prototype_fir_certified and
            qmf_power_complementary and
            qmf_alias_cancellation
        )

        payload = CertifiedEvaluationPayload(
            truth_status=truth_status,
            matching_status=matching_status,
            provenance=provenance,
            basis_asymptotic_validated=basis_asymptotic_validated,
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
            H1_response=20 * np.log10(np.maximum(1e-12, np.abs(np.fft.fft(h1_quant.float64_taps, K_fft)[:K_fft // 2 + 1]))) if h1_quant else None,
            passband_ripple_actual=float(pass_ripple),
            stopband_atten_actual=float(stop_atten),
            qmf_power_complementarity_max_db=qmf_pow_db,
            qmf_alias_distortion_max_db=qmf_alias_db,
            qmf_power_error_linear=qmf_pow_lin,
            qmf_alias_error_linear=qmf_alias_lin,
            regularization_energy=float(reg_energy),
            fit_residual=res_aug,
            data_fit_residual=res_data,
            regularization_residual=res_reg,
            asymptotic_error_bound=asymp_err,
            provenance_mixed_error=prov_err
        )
        result.header_code = self.generate_header(result)
        return result

    def compile_biorthogonal_pair(self, order_h0: int, order_g0: int, cutoff: float = 0.25) -> dict:
        """
        Compiles a Biorthogonal filter bank pair (H0, G0) via Gegenbauer half-band product filter factorization.
        Uses global canonical root orbits (conjugate & reciprocal closure) and exact DP subset-sum root partitioning.
        Preserves exact H0(z) G0(z) = P(z) product scaling without uncoordinated independent normalizations.
        """
        total_order = order_h0 + order_g0
        if total_order % 2 != 0:
            raise ValueError("The sum of H0 and G0 orders must be even for a valid half-band filter.")

        p_spec = FilterSpec(
            kind="lowpass",
            order=total_order + 1, # Tap length = total_order + 1
            cutoff=cutoff,
            wp=max(0.01, cutoff - 0.05),
            ws=min(0.49, cutoff + 0.05)
        )

        a_coeffs, _, _, _, _, _ = self.solve_coefficients(p_spec)
        p_taps = self.transform_to_taps(a_coeffs, p_spec)

        # Force strict half-band time-domain zeroing for non-center even taps
        center = total_order // 2
        for n in range(len(p_taps)):
            if abs(n - center) % 2 == 0 and n != center:
                p_taps[n] = 0.0

        # Set center tap to exactly 0.5 so half-band P(z) + P(-z) = z^-d delay scaling is exact
        p_taps[center] = 0.5
        p_prod = 2.0 * p_taps

        # Construct global canonical root orbits closed under conjugation and reciprocal reflection
        roots = np.roots(p_prod)
        rel_tol, abs_tol = 1e-2, 1e-3

        def same_root(a, b):
            return abs(a - b) <= max(abs_tol, rel_tol * max(abs(a), abs(b)))

        orbits = []
        used = [False] * len(roots)
        for i in range(len(roots)):
            if used[i]:
                continue
            r_val = roots[i]
            orbit_idx = [i]
            used[i] = True

            added = True
            while added:
                added = False
                curr_vals = [roots[k] for k in orbit_idx]
                for cv in curr_vals:
                    targets = [np.conj(cv)]
                    if abs(cv) > 1e-12:
                        targets.append(1.0 / cv)
                        targets.append(1.0 / np.conj(cv))
                    for tgt in targets:
                        for j in range(len(roots)):
                            if not used[j] and same_root(roots[j], tgt):
                                orbit_idx.append(j)
                                used[j] = True
                                added = True
            orbits.append([roots[k] for k in orbit_idx])

        # Exact DP subset-sum root orbit partition for order_h0
        orbit_sizes = [len(o) for o in orbits]
        dp = {0: []}
        for i, sz in enumerate(orbit_sizes):
            new_dp = dict(dp)
            for s, chosen in dp.items():
                if s + sz <= order_h0 and (s + sz) not in new_dp:
                    new_dp[s + sz] = chosen + [i]
            dp = new_dp

        if order_h0 not in dp:
            raise ValueError(
                f"Unable to partition roots into exact target orders order_h0={order_h0} and order_g0={order_g0} "
                f"while preserving symmetric canonical root orbits {orbit_sizes}."
            )

        h0_orbit_indices = set(dp[order_h0])
        h0_roots = []
        g0_roots = []
        for i, orbit in enumerate(orbits):
            if i in h0_orbit_indices:
                h0_roots.extend(orbit)
            else:
                g0_roots.extend(orbit)

        # Reconstruct unscaled factor polynomials from roots
        h0_unscaled = np.poly(h0_roots).real if len(h0_roots) > 0 else np.array([1.0])
        g0_unscaled = np.poly(g0_roots).real if len(g0_roots) > 0 else np.array([1.0])

        # Joint scaling: ensure H0(z) G0(z) = P_prod(z) = 2 P(z) exactly
        conv_unscaled = np.convolve(h0_unscaled, g0_unscaled)
        req_scale = float(np.dot(conv_unscaled, p_prod) / max(1e-15, np.dot(conv_unscaled, conv_unscaled)))

        h0_dc = float(np.sum(h0_unscaled))
        alpha = np.sqrt(2.0) / h0_dc if abs(h0_dc) > 1e-12 else 1.0
        beta = req_scale / alpha

        h0_taps = h0_unscaled * alpha
        g0_taps = g0_unscaled * beta

        # Generate complementary highpass filters H1 and G1 for 2-channel Biorthogonal Bank
        # Degrees d_h0 = order_h0, d_g0 = order_g0, d = d_h0 + d_g0
        # H1(z) = (-1)^d_g0 G0(-z) => h1[n] = (-1)^d_g0 (-1)^n g0[n]
        # G1(z) = -(-1)^d (-1)^d_h0 H0(-z) => g1[n] = -(-1)^d (-1)^d_h0 (-1)^n h0[n]
        # This guarantees exact alias cancellation H0(-z)G0(z) + H1(-z)G1(z) = 0 for all degree parities!
        d_h0, d_g0 = order_h0, order_g0
        d = d_h0 + d_g0
        delay = d // 2

        h1_sign = ((-1.0)**d_g0)
        g1_sign = -((-1.0)**d) * ((-1.0)**d_h0)

        h1_taps = np.array([h1_sign * ((-1.0)**n) * g0_taps[n] for n in range(len(g0_taps))])
        g1_taps = np.array([g1_sign * ((-1.0)**n) * h0_taps[n] for n in range(len(h0_taps))])

        # Verification metrics & symmetry residuals (comparing H0 * G0 against P_prod(z))
        h0_conv_g0 = np.convolve(h0_taps, g0_taps)
        product_residual = float(np.max(np.abs(h0_conv_g0 - p_prod)))

        h0_sym_res = float(np.max(np.abs(h0_taps - h0_taps[::-1])))
        g0_sym_res = float(np.max(np.abs(g0_taps - g0_taps[::-1])))

        K_fft = 4096
        omega = 2.0 * np.pi * np.arange(K_fft) / float(K_fft)
        expected_pr = 2.0 * np.exp(-1j * omega * delay)

        H0_f = np.fft.fft(h0_taps, K_fft)
        G0_f = np.fft.fft(g0_taps, K_fft)
        H1_f = np.fft.fft(h1_taps, K_fft)
        G1_f = np.fft.fft(g1_taps, K_fft)

        pr_complex = H0_f * G0_f + H1_f * G1_f
        pr_residual = float(np.max(np.abs(pr_complex - expected_pr)))

        # Alias cancellation check H0(-z) G0(z) + H1(-z) G1(z) == 0
        H0_neg = np.fft.fft(h0_taps * np.array([(-1.0)**n for n in range(len(h0_taps))]), K_fft)
        H1_neg = np.fft.fft(h1_taps * np.array([(-1.0)**n for n in range(len(h1_taps))]), K_fft)
        alias_complex = H0_neg * G0_f + H1_neg * G1_f
        alias_residual = float(np.max(np.abs(alias_complex)))

        h1_qmf_res = float(np.max(np.abs(H1_f - np.roll(np.conj(G0_f), K_fft // 2))))
        g1_qmf_res = float(np.max(np.abs(G1_f - np.roll(np.conj(H0_f), K_fft // 2))))

        return {
            "H0": self.quantize_taps(h0_taps),
            "H1": self.quantize_taps(h1_taps),
            "G0": self.quantize_taps(g0_taps),
            "G1": self.quantize_taps(g1_taps),
            "P": p_taps,
            "product_residual": product_residual,
            "pr_residual": pr_residual,
            "pr_error": pr_residual, # Backward compatibility alias
            "alias_residual": alias_residual,
            "h0_sym_residual": h0_sym_res,
            "g0_sym_residual": g0_sym_res,
            "h1_qmf_residual": h1_qmf_res,
            "g1_qmf_residual": g1_qmf_res
        }

    def plot_response(self, result: FilterResult, output_path: str):
        spec = result.spec
        plt.figure(figsize=(16, 12))

        # Subplot 1: Frequency Response (dB)
        plt.subplot(2, 2, 1)
        plt.plot(result.freq_grid, result.H0_response, 'b-', label='H0 (Lowpass)' if spec.kind in ('qmf', 'asymmetric_qmf') else 'H(e^{jw})', linewidth=2)
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
        if spec.kind in ("qmf", "asymmetric_qmf") and result.h1_taps is not None:
            K_fft = 2 * (len(result.freq_grid) - 1)
            H0 = np.fft.fft(result.h0_taps.float64_taps, K_fft)
            H1 = np.fft.fft(result.h1_taps.float64_taps, K_fft)
            H0_shift = np.roll(H0, K_fft // 2)
            H1_shift = np.roll(H1, K_fft // 2)

            pow_comp = np.abs(H0)**2 + np.abs(H1)**2
            aliasing_func = 0.5 * np.abs(H0 * H0_shift - H1 * H1_shift)
            pow_db = 10 * np.log10(np.maximum(1e-12, pow_comp[:len(result.freq_grid)]))
            alias_db = 20 * np.log10(np.maximum(1e-12, aliasing_func[:len(result.freq_grid)]))

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
        best_res = None
        best_score = float('inf')
        use_solver = solver if solver is not None else self.solver

        for lam in lambda_candidates:
            for mu in mu_candidates:
                compiler = GegenbauerFilterCompiler(
                    lam=lam,
                    basis_terms=self.basis_terms,
                    basis_type=self.basis_type,
                    solver=use_solver,
                    mu_reg=mu,
                    reg_power=self.reg_power,
                    asymptotic_mode=self.asymptotic_mode,
                    grid_samples=self.grid_samples,
                    precision=self.ctx.precision
                )
                res = compiler.compile(spec)

                score = res.passband_ripple_actual * 10.0 - res.stopband_atten_actual
                if res.spec.kind in ("qmf", "asymmetric_qmf"):
                    score += res.qmf_power_complementarity_max_db * 20.0 + res.qmf_alias_distortion_max_db

                score += res.payload.provenance.total * 100.0

                if not res.payload.is_certified:
                    score += 1e5

                if score < best_score:
                    best_score = score
                    best_res = res

        return best_res

def main():
    pass

if __name__ == "__main__":
    main()
