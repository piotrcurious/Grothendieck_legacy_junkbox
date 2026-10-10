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


class FactorMode(Enum):
    """Spectral Factorization Backends for Half-band Power Polynomial P(z)."""
    CEPSTRAL_APPROX = "CEPSTRAL_APPROX"           # Production cepstral log-spectrum minimum-phase factorization
    STRUCTURED_CHEBYSHEV = "STRUCTURED_CHEBYSHEV" # Chebyshev x-domain root lifting in x = cos(w)
    REFERENCE_ROOTS = "REFERENCE_ROOTS"           # Reference monomial degree-2M root finding backend


class VerificationStatus(Enum):
    """Tri-state verification status for layer-wise filter certification."""
    NOT_APPLICABLE = "NOT_APPLICABLE"
    VERIFIED = "VERIFIED"
    FAILED = "FAILED"


class WarrantDisposition(Enum):
    """Disposition status of an epistemic claim warrant."""
    ESTABLISHED = "ESTABLISHED"
    CONTRADICTED = "CONTRADICTED"
    OPEN = "OPEN"
    NOT_EVALUATED = "NOT_EVALUATED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class EvidenceMethod(Enum):
    """Evidence method licensing an epistemic claim."""
    ALGEBRAIC_PROOF = "ALGEBRAIC_PROOF"
    EXACT_ARITHMETIC = "EXACT_ARITHMETIC"
    ANALYTIC_BOUND = "ANALYTIC_BOUND"
    VALIDATED_NUMERICAL_BOUND = "VALIDATED_NUMERICAL_BOUND"
    ORDINARY_NUMERICAL_DIAGNOSTIC = "ORDINARY_NUMERICAL_DIAGNOSTIC"
    UNPROVED_SCHEMA = "UNPROVED_SCHEMA"


@dataclass
class ClaimWarrant:
    """Epistemic warrant tracking proposition, evidence method, disposition, and validity bounds."""
    proposition: str
    target_type: str
    domain: str
    disposition: WarrantDisposition
    method: EvidenceMethod
    evidence_payload: dict = field(default_factory=dict)
    assumptions: List[str] = field(default_factory=list)
    invalidating_conditions: List[str] = field(default_factory=list)


@dataclass
class EpistemicContract:
    """Contract preserving claims, warrants, required scope, and evidence graph for a computational target."""
    target_name: str
    target_type: str
    required_claims: List[str] = field(default_factory=list)
    warrants: List[ClaimWarrant] = field(default_factory=list)

    def add_warrant(self, warrant: ClaimWarrant) -> None:
        self.warrants.append(warrant)

    def get_warrant(self, proposition: str) -> Optional[ClaimWarrant]:
        for w in self.warrants:
            if w.proposition == proposition:
                return w
        return None

    def missing_claims(self) -> List[str]:
        present = {w.proposition for w in self.warrants}
        return [claim for claim in self.required_claims if claim not in present]

    def unresolved_claims(self) -> List[str]:
        by_prop = {w.proposition: w for w in self.warrants}
        unresolved = []
        for claim in self.required_claims:
            w = by_prop.get(claim)
            if w is None or w.disposition != WarrantDisposition.ESTABLISHED:
                unresolved.append(claim)
        return unresolved

    @property
    def is_fully_established(self) -> bool:
        if len(self.required_claims) > 0:
            return len(self.unresolved_claims()) == 0
        return len(self.warrants) > 0 and all(w.disposition == WarrantDisposition.ESTABLISHED for w in self.warrants)


class PropagationMode(Enum):
    """Compositional Error Propagation Behavior for Transformation Steps."""
    EXACT_MORPHISM = "EXACT_MORPHISM"         # L_i = 1.0 (exact isometry/morphism)
    NORM_PRESERVING = "NORM_PRESERVING"       # L_i = 1.0 (unitary transform e.g. T_lambda S_lambda)
    LIPSCHITZ_BOUNDED = "LIPSCHITZ_BOUNDED"   # L_i > 0.0 (Lipschitz continuous mapping)
    UNESTABLISHED = "UNESTABLISHED"           # Propagation constant not established


@dataclass
class TransformationContract:
    """Transformation contract tracking staged perturbation chain F_{i+1} = Phi_i(F_i) + delta_i."""
    stage_name: str
    input_target_type: str
    output_target_type: str
    propagation_mode: PropagationMode
    lipschitz_constant: float = 1.0
    stage_error_bound: float = 0.0
    input_contract: Optional[EpistemicContract] = None
    output_contract: Optional[EpistemicContract] = None

    def __post_init__(self):
        if self.propagation_mode in (PropagationMode.EXACT_MORPHISM, PropagationMode.NORM_PRESERVING):
            self.lipschitz_constant = 1.0
        if self.lipschitz_constant < 0.0 or not np.isfinite(self.lipschitz_constant):
            raise ValueError(f"Lipschitz constant must be finite and non-negative, got {self.lipschitz_constant}")
        if self.stage_error_bound < 0.0 or not np.isfinite(self.stage_error_bound):
            raise ValueError(f"Stage error bound must be finite and non-negative, got {self.stage_error_bound}")

    def propagate_error(self, incoming_error: float) -> float:
        if self.propagation_mode == PropagationMode.UNESTABLISHED:
            return float('inf')
        if not np.isfinite(incoming_error) or incoming_error < 0.0:
            return float('inf')
        return self.lipschitz_constant * incoming_error + self.stage_error_bound


@dataclass
class RootOrbit:
    """Algebraic Root Orbit Unit for conjugate/reciprocal symmetry decomposition."""
    kind: str  # REAL_ENDPOINT, REAL_RECIPROCAL_PAIR, UNIT_CIRCLE_PAIR, COMPLEX_RECIPROCAL_CONJUGATE_QUARTET
    roots: np.ndarray
    multiplicity: int
    orbit_residual: float

    @property
    def size(self) -> int:
        return len(self.roots)


class RootDecomposer:
    """Shared Multiplicity-Aware Conjugate & Reciprocal Root Orbit Decomposition Engine."""

    @staticmethod
    def decompose_halfband_roots(p_taps: np.ndarray, total_z_degree: int, rel_tol: float = 1e-4, abs_tol: float = 1e-5) -> List[RootOrbit]:
        center = (len(p_taps) - 1) // 2
        M_prod = total_z_degree // 2
        b_harmonics = np.zeros(M_prod + 1, dtype=np.float64)
        b_harmonics[0] = p_taps[center]
        for m in range(1, M_prod + 1):
            if center + m < len(p_taps):
                b_harmonics[m] = 2.0 * p_taps[center + m]

        x_roots = np.polynomial.chebyshev.chebroots(b_harmonics)
        roots = []
        for x_val in x_roots:
            disc = np.sqrt(x_val**2 - 1.0 + 0j)
            roots.extend([x_val - disc, x_val + disc])

        def same_root(a, b):
            return abs(a - b) <= max(abs_tol, rel_tol * max(abs(a), abs(b)))

        raw_clusters = []
        used_r = [False] * len(roots)
        for i in range(len(roots)):
            if used_r[i]:
                continue
            group = [i]
            used_r[i] = True
            for j in range(i + 1, len(roots)):
                if not used_r[j] and same_root(roots[j], roots[i]):
                    group.append(j)
                    used_r[j] = True
            rep = np.mean([roots[k] for k in group])
            raw_clusters.append({'rep': rep, 'mult': len(group)})

        used_c = [False] * len(raw_clusters)
        atomic_units = []

        for i in range(len(raw_clusters)):
            if used_c[i]:
                continue
            c1 = raw_clusters[i]
            r1 = c1['rep']
            m1 = c1['mult']
            used_c[i] = True

            if abs(np.imag(r1)) < 1e-3:
                r1_real = float(np.real(r1))
                if abs(abs(r1_real) - 1.0) < 1e-3:
                    ep_val = 1.0 if r1_real > 0 else -1.0
                    atomic_units.append(RootOrbit(
                        kind='REAL_ENDPOINT',
                        roots=np.array([ep_val]),
                        multiplicity=m1,
                        orbit_residual=abs(r1_real - ep_val)
                    ))
                else:
                    recip_idx = -1
                    for j in range(len(raw_clusters)):
                        if not used_c[j] and abs(raw_clusters[j]['rep'] - 1.0 / r1_real) < 1e-2:
                            recip_idx = j
                            break
                    if recip_idx != -1:
                        c2 = raw_clusters[recip_idx]
                        if m1 != c2['mult']:
                            raise ValueError(f"Reciprocal orbit multiplicities disagree for root {r1_real:.6f}: {m1} vs {c2['mult']}.")
                        used_c[recip_idx] = True
                        atomic_units.append(RootOrbit(
                            kind='REAL_RECIPROCAL_PAIR',
                            roots=np.array([r1_real, float(np.real(c2['rep']))]),
                            multiplicity=m1,
                            orbit_residual=abs(r1_real * c2['rep'] - 1.0)
                        ))
                    else:
                        raise ValueError(f"Unpaired real reciprocal root detected: {r1_real:.6f} without reciprocal partner.")
            else:
                is_unit = abs(abs(r1) - 1.0) < 1e-3
                if is_unit:
                    conj_idx = -1
                    for j in range(len(raw_clusters)):
                        if not used_c[j] and abs(raw_clusters[j]['rep'] - np.conj(r1)) < 1e-2:
                            conj_idx = j
                            break
                    if conj_idx != -1:
                        c2 = raw_clusters[conj_idx]
                        if m1 != c2['mult']:
                            raise ValueError(f"Unit circle conjugate orbit multiplicities disagree for root {r1}: {m1} vs {c2['mult']}.")
                        used_c[conj_idx] = True
                        unit_roots = np.array([r1, c2['rep']])
                    else:
                        raise ValueError(f"Unpaired unit circle conjugate root detected for {r1}: conjugate partner not found.")
                    atomic_units.append(RootOrbit(
                        kind='UNIT_CIRCLE_PAIR',
                        roots=unit_roots,
                        multiplicity=m1,
                        orbit_residual=abs(abs(r1) - 1.0)
                    ))
                else:
                    quad_indices = [i]
                    for j in range(len(raw_clusters)):
                        if not used_c[j]:
                            r2 = raw_clusters[j]['rep']
                            if abs(r2 - np.conj(r1)) < 1e-2 or abs(r2 - 1.0 / r1) < 1e-2 or abs(r2 - 1.0 / np.conj(r1)) < 1e-2:
                                quad_indices.append(j)
                                used_c[j] = True
                    multiplicities = [raw_clusters[k]['mult'] for k in quad_indices]
                    if len(set(multiplicities)) != 1:
                        raise ValueError(f"Complex root orbit multiplicities disagree: {multiplicities}")
                    quad_mult = multiplicities[0]
                    quad_roots = np.array([raw_clusters[k]['rep'] for k in quad_indices])
                    if len(quad_roots) != 4:
                        raise ValueError(f"Invalid root orbit size for COMPLEX_RECIPROCAL_CONJUGATE_QUARTET: got {len(quad_roots)}, expected 4.")
                    # Compute actual orbit residual: max discrepancy from exact conjugate/reciprocal relations
                    r_conj = np.conj(r1)
                    r_recip = 1.0 / r1
                    r_conj_recip = 1.0 / np.conj(r1)
                    res_quad = float(max(
                        min(abs(r - r_conj) for r in quad_roots),
                        min(abs(r - r_recip) for r in quad_roots),
                        min(abs(r - r_conj_recip) for r in quad_roots)
                    ))
                    atomic_units.append(RootOrbit(
                        kind='COMPLEX_RECIPROCAL_CONJUGATE_QUARTET',
                        roots=quad_roots,
                        multiplicity=quad_mult,
                        orbit_residual=res_quad
                    ))

        orbit_total_size = sum(unit.size * unit.multiplicity for unit in atomic_units)
        if orbit_total_size != total_z_degree:
            raise ValueError(f"Root-orbit decomposition is not closed: got {orbit_total_size} roots, expected {total_z_degree}.")

        return atomic_units


@dataclass
class FactorizationDiagnostics:
    """Layer VIII Spectral Factorization Quality & Invariant Diagnostics."""
    mode: FactorMode
    target_degree: int
    actual_degree: int
    coefficient_residual_abs: float
    coefficient_residual_rel: float
    frequency_residual_abs: float
    frequency_residual_rel: float
    factor_imaginary_residual: float
    positivity_min: float
    positivity_max: float
    halfband_residual: float
    zero_on_unit_circle_count: int
    finite: bool
    certified: bool

    @property
    def coefficient_residual(self) -> float:
        return self.coefficient_residual_abs

    @property
    def frequency_residual(self) -> float:
        return self.frequency_residual_abs

    @property
    def factor_degree(self) -> int:
        return self.actual_degree


@dataclass
class ErrorEstimateProvenance:
    """Layer VIII Non-Double-Counting Error Estimate Decomposition Payload."""
    e_analytic: float = 0.0
    e_arithmetic: float = 0.0
    e_conditioning: float = 0.0
    e_implementation: float = 0.0

    @property
    def total(self) -> float:
        return self.e_analytic + self.e_arithmetic + self.e_conditioning + self.e_implementation

ErrorBoundProvenance = ErrorEstimateProvenance


@dataclass
class CertifiedEvaluationPayload:
    """Layer VIII Certified Evaluation Payload separating diagnostics from certified bounds."""
    truth_status: TruthStatus
    matching_status: MatchingStatus
    provenance: ErrorEstimateProvenance
    basis_asymptotic: VerificationStatus = VerificationStatus.NOT_APPLICABLE
    prototype_fir: VerificationStatus = VerificationStatus.NOT_APPLICABLE
    qmf_power: VerificationStatus = VerificationStatus.NOT_APPLICABLE
    qmf_alias: VerificationStatus = VerificationStatus.NOT_APPLICABLE
    factorization: VerificationStatus = VerificationStatus.NOT_APPLICABLE
    structural_certified: bool = False
    specification_met: bool = False
    asymptotic_diagnostic_passed: bool = False
    is_certified: bool = False

    @property
    def basis_asymptotic_validated(self) -> bool:
        return self.basis_asymptotic == VerificationStatus.VERIFIED

    @property
    def prototype_fir_certified(self) -> bool:
        return self.prototype_fir == VerificationStatus.VERIFIED

    @property
    def qmf_power_complementary(self) -> bool:
        return self.qmf_power == VerificationStatus.VERIFIED

    @property
    def qmf_alias_cancellation(self) -> bool:
        return self.qmf_alias == VerificationStatus.VERIFIED

    @property
    def factorization_certified(self) -> bool:
        return self.factorization == VerificationStatus.VERIFIED


def _fit_spectral_factor_scale(q: np.ndarray, p_taps: np.ndarray, tol: float = 1e-15, imag_tol: float = 1e-5) -> np.ndarray:
    """Least-squares optimal scalar recovery scaling for spectral factor q such that r = q * q_rev fits p_taps."""
    q_arr = np.asarray(q)
    if not np.all(np.isfinite(q_arr)):
        raise ValueError("Non-finite spectral-factor candidate.")

    imag_res = float(np.max(np.abs(np.imag(q_arr)))) if np.iscomplexobj(q_arr) else 0.0
    if imag_res > imag_tol:
        raise ValueError(f"Candidate is not real within tolerance: imaginary residual {imag_res:.3e} exceeds {imag_tol:.3e}.")

    q_real = np.real(q_arr).astype(np.float64)
    r = np.convolve(q_real, q_real[::-1])
    den = float(np.vdot(r, r).real)
    if den <= tol:
        raise ValueError("Degenerate spectral factor candidate.")
    s = float(np.vdot(r, p_taps).real / den)
    if s <= 0.0:
        raise ValueError(f"Spectral-factor scalar fit is non-positive: s={s:.6e}")
    return q_real * math.sqrt(s)


def determine_truth_status(lam: float) -> TruthStatus:
    if abs(lam) < 1e-12:
        raise ValueError("Gegenbauer parameter lambda=0 is a singular limit; use Chebyshev polynomial limit T_n(x).")
    if -0.5 < lam < 0.0:
        return TruthStatus.LIMIT_CIRCLE_SUBCRITICAL
    if lam > 0 and abs(2.0 * lam - round(2.0 * lam)) < 1e-12:
        return TruthStatus.PHYSICAL_SPHERE_GEOMETRY
    return TruthStatus.ANALYTIC_CONTINUATION


def gegenbauer_to_cosine_matrix(max_degree: int, lam: float, normalized: bool = True) -> np.ndarray:
    """
    Computes exact algebraic Gegenbauer-to-Cosine expansion matrix C_cos[n, m]
    mapping Gegenbauer basis functions phi_n^(lambda)(cos w) or C_n^(lambda)(cos w) to cosine Fourier harmonics cos(m w).
    """
    C_cos = np.zeros((max_degree + 1, max_degree + 1), dtype=np.float64)
    C_cos[0, 0] = 1.0
    if max_degree >= 1:
        C_cos[1, 1] = 2.0 * float(lam)

    for deg in range(2, max_degree + 1):
        alpha_d = 2.0 * (deg + lam - 1.0) / float(deg)
        beta_d = (deg + 2.0 * lam - 2.0) / float(deg)
        x_C1 = np.zeros(max_degree + 1, dtype=np.float64)
        for m in range(deg):
            val = C_cos[deg - 1, m]
            if val != 0.0:
                if m == 0:
                    x_C1[1] += val
                else:
                    x_C1[m + 1] += 0.5 * val
                    x_C1[abs(m - 1)] += 0.5 * val
        C_cos[deg, :] = alpha_d * x_C1 - beta_d * C_cos[deg - 2, :]

    if normalized:
        for deg in range(max_degree + 1):
            c1 = float(c_n_1_val(deg, lam))
            if abs(c1) > 1e-12:
                C_cos[deg, :] /= c1

    return C_cos


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
            if self.wp + self.ws > 0.5 + 1e-6:
                raise ValueError(f"Asymmetric QMF target transition edges must satisfy wp + ws <= 0.5 to remain compatible with half-band complementarity, got wp={self.wp}, ws={self.ws} (sum = {self.wp + self.ws})")

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
class GegenbauerSpectralDesign:
    """Primary Layer VIII Mathematical Representation of Gegenbauer Spectral Multiplier F(x) = sum a_n phi_n(x)."""
    lam: float
    a_coeffs: np.ndarray
    basis_degrees: np.ndarray
    basis_terms: int
    operator_eigenvalues: np.ndarray
    sturm_liouville_energy: float
    projection_residual: float
    quadrature_residual: float
    conditioning: float
    asymptotic_sampled_discrepancy: float
    truth_status: TruthStatus
    matching_status: MatchingStatus
    augmented_design_residual: float = 0.0
    data_design_residual: float = 0.0
    regularization_residual: float = 0.0
    final_realization_residual: float = 0.0

    @property
    def asymptotic_error_bound(self) -> float:
        return self.asymptotic_sampled_discrepancy


@dataclass
class FilterResult:
    """Compiler output containing taps, metrics, certifications, payloads, and headers."""
    spec: FilterSpec
    lam: float
    basis_terms: int
    mu_reg: float
    h0_taps: QuantizedTaps
    payload: CertifiedEvaluationPayload
    epistemic_contract: Optional[EpistemicContract] = None
    factorization: Optional[FactorizationDiagnostics] = None
    h1_taps: Optional[QuantizedTaps] = None
    spectral_design: Optional[GegenbauerSpectralDesign] = None
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
    augmented_design_residual: float = 0.0
    data_design_residual: float = 0.0
    final_realization_residual: float = 0.0
    asymptotic_sampled_discrepancy: float = 0.0
    provenance_mixed_error: float = 0.0

    @property
    def asymptotic_error_bound(self) -> float:
        return self.asymptotic_sampled_discrepancy
    header_code: str = ""

    def summary(self) -> str:
        lines = [
            "=== Gegenbauer Filter Compiler Execution Summary ===",
            f"Filter Type: {self.spec.kind.upper()} | Order N: {self.spec.order} | Cutoff: {self.spec.cutoff} fs",
            f"Gegenbauer Lambda: {self.lam:.4f} | Basis Terms: {self.basis_terms}",
            f"Truth Status Topology: {self.payload.truth_status.value}",
            f"Asymptotic Matching Certification: {self.payload.matching_status.value}",
            f"Certifications: Basis={self.payload.basis_asymptotic.value} | Prototype={self.payload.prototype_fir.value}"
            f" | QMF Power={self.payload.qmf_power.value} | QMF Alias={self.payload.qmf_alias.value}"
            f" | Factorization={self.payload.factorization.value} | Total={self.payload.is_certified}",
            f"Passband Ripple: {self.passband_ripple_actual:.4f} dB | Stopband Attenuation: {self.stopband_atten_actual:.2f} dB",
        ]
        if self.factorization is not None:
            lines.append(f"Factorization Mode: {self.factorization.mode.value} | Coeff Residual: {self.factorization.coefficient_residual:.6e} | Freq Residual: {self.factorization.frequency_residual:.6e}")
        if self.spec.kind in ("qmf", "asymmetric_qmf"):
            lines.append(f"QMF Power Complementarity Peak Ripple: {self.qmf_power_complementarity_max_db:.4f} dB")
            lines.append(f"QMF Peak Alias Distortion: {self.qmf_alias_distortion_max_db:.2f} dB")
        lines.append(f"Sturm-Liouville Regularization Energy: {self.regularization_energy:.6e}")
        lines.append(f"Asymptotic Sampled Discrepancy (E_analytic): {self.payload.provenance.e_analytic:.6e}")
        lines.append(f"Fixed-Point Quantization Noise (E_arithmetic): {self.payload.provenance.e_arithmetic:.6e}")
        lines.append(f"Total Error Estimate (E_total_estimate): {self.payload.provenance.total:.6e}")
        return "\n".join(lines)


class WarrantedBoundSelector:
    """Policy-based decision layer and release gate evaluating warranted error bounds across candidate results."""

    @staticmethod
    def select_optimal_warranted_candidate(
        candidates: List[FilterResult],
        require_established_targets: bool = True
    ) -> Tuple[Optional[FilterResult], str]:
        if not candidates:
            return None, "NO_CANDIDATES"

        eligible = []
        for cand in candidates:
            contract = cand.epistemic_contract
            if contract is None:
                continue

            if require_established_targets and not contract.is_fully_established:
                continue

            real_warrant = contract.get_warrant("REALIZATION_VERIFIED")
            if real_warrant is None or real_warrant.disposition != WarrantDisposition.ESTABLISHED:
                continue

            # Score by final realization residual and mixed error
            score = cand.final_realization_residual + 0.1 * cand.provenance_mixed_error
            eligible.append((cand, score))

        if not eligible:
            return None, "NO_WARRANTED_BOUND"

        eligible.sort(key=lambda x: x[1])
        best_cand, best_err = eligible[0]
        return best_cand, f"OPTIMAL_WARRANTED_BOUND_{best_err:.6e}"


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
        factor_mode: FactorMode = FactorMode.STRUCTURED_CHEBYSHEV,
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
        self.factor_mode = factor_mode
        self.grid_samples = grid_samples
        if precision != PrecisionType.FLOAT64:
            raise NotImplementedError("Only PrecisionType.FLOAT64 execution is currently implemented.")
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

    def _eval_basis(self, n: int, x: np.ndarray) -> np.ndarray:
        """Evaluates pure n-th Gegenbauer basis function phi_n^(lambda)(x) at array x in [-1, 1]."""
        x_arr = np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0)
        if self.basis_type == "normalized":
            return normalized_phi_recurrence(n, self.lam, x_arr)
        else:
            c1 = float(c_n_1_val(n, self.lam))
            return c1 * normalized_phi_recurrence(n, self.lam, x_arr)

    def _apply_fir_symmetry_envelope(self, P_k: np.ndarray, x: np.ndarray, symmetry: SymmetryClass) -> np.ndarray:
        """Applies FIR Type I-IV symmetry envelope modulation to Gegenbauer basis evaluation."""
        x_arr = np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0)
        if symmetry == SymmetryClass.TYPE_I:
            return P_k
        elif symmetry == SymmetryClass.TYPE_II:
            return np.sqrt(0.5 * (1.0 + x_arr)) * P_k
        elif symmetry == SymmetryClass.TYPE_III:
            return np.sqrt(np.maximum(0.0, 1.0 - x_arr**2)) * P_k
        elif symmetry == SymmetryClass.TYPE_IV:
            return np.sqrt(np.maximum(0.0, 0.5 * (1.0 - x_arr))) * P_k
        return P_k

    def _eval_fir_basis(self, n: int, x: np.ndarray, symmetry: SymmetryClass = SymmetryClass.TYPE_I) -> np.ndarray:
        """Evaluates n-th Gegenbauer basis function with FIR Type I-IV symmetry envelope for tap realization."""
        phi_k = self._eval_basis(n, x)
        return self._apply_fir_symmetry_envelope(phi_k, x, symmetry)

    def _eval_pure_asymptotic_basis(self, n: int, omega: np.ndarray) -> np.ndarray:
        """
        Evaluates pure Gegenbauer asymptotic approximation phi_n^asymp(theta) without FIR symmetry envelopes.
        """
        theta = omega
        if self.asymptotic_mode == "bessel":
            return endpoint_bessel_leading(n, self.lam, theta)
        elif self.asymptotic_mode == "wkb":
            return interior_wkb_approx(n, self.lam, theta)
        elif self.asymptotic_mode == "composite" or self.asymptotic_mode == "auto":
            return composite_matched_approx(n, self.lam, theta)
        else:
            return composite_matched_approx(n, self.lam, theta)

    def _eval_asymptotic_basis(self, n: int, omega: np.ndarray, symmetry: SymmetryClass = SymmetryClass.TYPE_I) -> np.ndarray:
        """Evaluates Gegenbauer asymptotic approximation with FIR Type I-IV symmetry envelope modulation."""
        phi_asymp = self._eval_pure_asymptotic_basis(n, omega)
        x = np.cos(omega)
        return self._apply_fir_symmetry_envelope(phi_asymp, x, symmetry)

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

    def solve_parity_constrained_gegenbauer_response(self, spec: FilterSpec) -> Tuple[np.ndarray, np.ndarray, int, float, float, float, float]:
        """
        Solves for parity-constrained Gegenbauer response P_lambda(x) = 0.5 + R_odd(x) in odd Gegenbauer basis C_{2k+1}^(\\lambda)(x).
        Structurally guarantees P_lambda(x) + P_lambda(-x) = 1 in x = cos(omega) space via Gegenbauer odd parity C_{2k+1}^(\\lambda)(-x) = -C_{2k+1}^(\\lambda)(x).
        Returns (a_coeffs, p_taps, K, cond_val, res_aug, res_data, res_reg).
        """
        N = spec.order       # N taps for H0
        M = N - 1            # Degree of H0 = N - 1
        center = M           # Center tap index of P(z) = M (length 2M + 1)
        K_full = (M + 1) // 2
        K = min(K_full, self.basis_terms) if self.basis_terms is not None else K_full

        nodes, weights = gauss_gegenbauer_quadrature(self.grid_samples, self.lam)
        omega_q = np.arccos(nodes)

        D_q, W_q = self._build_spectral_target(spec, omega_q)
        R_target = np.abs(D_q)**2 - 0.5

        A_odd = np.zeros((self.grid_samples, K))
        for k in range(K):
            n_odd = 2 * k + 1
            A_odd[:, k] = self._eval_basis(n_odd, nodes)

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

        # Convert R_odd(cos w) to Fourier cosine taps b_k via exact Gegenbauer-to-Cosine expansion
        p_taps = np.zeros(2 * M + 1, dtype=np.float64)
        p_taps[center] = 0.5 # Center tap corresponds to constant 0.5

        max_deg = 2 * K
        C_cos = gegenbauer_to_cosine_matrix(max_deg, self.lam, normalized=(self.basis_type == "normalized"))

        b_harmonics = np.zeros(max_deg + 1, dtype=np.float64)
        for k in range(K):
            n_odd = 2 * k + 1
            b_harmonics += a_odd[k] * C_cos[n_odd, :]

        for m in range(1, min(M + 1, max_deg + 1)):
            if m % 2 != 0:
                b_m = b_harmonics[m]
                p_taps[center - m] = b_m / 2.0
                p_taps[center + m] = b_m / 2.0

        # Ensure P(w) = 0.5 + R_odd(w) is non-negative P(w) >= 0 and P(w) <= 1
        w_eval = np.linspace(0, np.pi, 2048)
        P_w = np.zeros_like(w_eval)
        for n, val in enumerate(p_taps):
            P_w += val * np.cos((n - center) * w_eval)

        min_P = np.min(P_w)
        max_P = np.max(P_w)
        over = max(0.0, -min_P, max_P - 1.0)
        if over > 1e-12:
            scale_factor = 0.5 / (0.5 + over)
            a_odd *= scale_factor
            a_coeffs *= scale_factor
            res_data = float(np.linalg.norm(A_w @ a_odd - D_w) / max(np.linalg.norm(D_w), 1e-15))
            res_reg = float(np.linalg.norm(R_mat @ a_odd) / max(np.linalg.norm(D_w), 1e-15))
            res_aug = float(np.linalg.norm(np.concatenate([A_w @ a_odd - D_w, R_mat @ a_odd])) / max(np.linalg.norm(D_sys), 1e-15))
            for m in range(1, M + 1):
                if m % 2 != 0:
                    p_taps[center - m] *= scale_factor
                    p_taps[center + m] *= scale_factor

        return a_coeffs, p_taps, K, cond_val, res_aug, res_data, res_reg

    def solve_halfband_power_polynomial(self, spec: FilterSpec) -> Tuple[np.ndarray, np.ndarray, int, float, float, float, float]:
        """Backward-compatibility alias for solve_parity_constrained_gegenbauer_response."""
        return self.solve_parity_constrained_gegenbauer_response(spec)

    def validate_power_polynomial(self, p_taps: np.ndarray, grid_size: int = 2048) -> Tuple[bool, float, float, float]:
        """
        Validates half-band power polynomial P(z) using numerical critical point evaluation.
        Evaluates P(x) at endpoints x = +-1 and all real derivative roots P'(x) = 0 in [-1, 1].
        Returns (is_valid, min_P, max_P, max_halfband_err).
        """
        center = (len(p_taps) - 1) // 2
        M = center

        # First check coefficient symmetry p_taps[center-m] == p_taps[center+m]
        sym_err = float(np.max(np.abs(p_taps[:center] - p_taps[center+1:][::-1]))) if center > 0 else 0.0

        b_harmonics = np.zeros(M + 1, dtype=np.float64)
        b_harmonics[0] = p_taps[center]
        for m in range(1, M + 1):
            if center + m < len(p_taps):
                b_harmonics[m] = 2.0 * p_taps[center + m]

        # Compute Chebyshev derivative P'(x)
        b_der = np.polynomial.chebyshev.chebder(b_harmonics)
        crit_pts = [-1.0, 1.0]

        if len(b_der) > 0:
            crit_roots = np.polynomial.chebyshev.chebroots(b_der)
            for r in crit_roots:
                if abs(np.imag(r)) < 1e-6 and -1.0 <= r.real <= 1.0:
                    crit_pts.append(float(r.real))

        # Evaluate P(x) at endpoints and critical points
        vals = np.polynomial.chebyshev.chebval(crit_pts, b_harmonics)

        # Dense grid check to verify critical point extrema
        if grid_size > 0:
            x_dense = np.linspace(-1.0, 1.0, grid_size)
            vals_dense = np.polynomial.chebyshev.chebval(x_dense, b_harmonics)
            min_P = min(float(np.min(vals)), float(np.min(vals_dense)))
            max_P = max(float(np.max(vals)), float(np.max(vals_dense)))
        else:
            min_P = float(np.min(vals))
            max_P = float(np.max(vals))

        # Complementarity residual bound: |2 b0 - 1| + 2 * sum_{m>0, even} |b_m|
        b_even_sum = 2.0 * float(np.sum(np.abs(b_harmonics[2::2]))) if len(b_harmonics) > 2 else 0.0
        max_hb_err = float(abs(2.0 * b_harmonics[0] - 1.0) + b_even_sum + sym_err)

        tau_P = 1e-4 * max(1.0, abs(max_P))
        is_valid = (min_P >= -tau_P) and (max_hb_err <= 1e-4)
        return is_valid, min_P, max_P, max_hb_err

    def _compute_factorization_diagnostics(
        self,
        h0_complex: np.ndarray,
        p_taps: np.ndarray,
        mode: FactorMode,
        target_degree: Optional[int] = None
    ) -> Tuple[np.ndarray, FactorizationDiagnostics]:
        imag_res = float(np.max(np.abs(np.imag(h0_complex)))) if np.iscomplexobj(h0_complex) else 0.0
        if imag_res > 1e-5:
            raise ValueError(f"Factorization produced non-real filter taps: imaginary residual {imag_res:.3e} exceeds 1e-5 tolerance.")

        h0 = np.real(h0_complex)
        M_target = target_degree if target_degree is not None else len(h0) - 1
        center = (len(p_taps) - 1) // 2

        # Validate half-band power polynomial P(z)
        _, min_P, max_P, hb_err = self.validate_power_polynomial(p_taps)

        r_fact = np.convolve(h0, h0[::-1])
        coeff_res_abs = float(np.max(np.abs(r_fact - p_taps)))
        coeff_res_rel = coeff_res_abs / max(1.0, float(np.max(np.abs(p_taps))))

        K_fft = 4096
        w = 2.0 * np.pi * np.arange(K_fft) / float(K_fft)
        P_w = np.zeros(K_fft, dtype=np.float64)
        for n, val in enumerate(p_taps):
            P_w += val * np.cos((n - center) * w)

        H0_f = np.fft.fft(h0, K_fft)
        H0_pow = np.abs(H0_f)**2
        freq_res_abs = float(np.max(np.abs(H0_pow - P_w)))
        freq_res_rel = freq_res_abs / max(1.0, float(max_P))

        non_zero = np.where(np.abs(h0) > 1e-12)[0]
        if len(non_zero) == 0:
            actual_deg = -1
        elif len(non_zero) == 1:
            actual_deg = 0
        else:
            actual_deg = int(non_zero[-1] - non_zero[0])

        # Count zeros near unit circle |z - 1| < 1e-3
        roots_h0 = np.roots(h0) if len(h0) > 1 else np.array([])
        zero_uc_count = int(np.sum(np.abs(np.abs(roots_h0) - 1.0) < 1e-3))

        is_finite = bool(np.all(np.isfinite(h0)) and np.isfinite(coeff_res_abs) and np.isfinite(freq_res_abs))
        certified = self.certify_factorization_diagnostics(
            is_finite=is_finite,
            actual_degree=actual_deg,
            target_degree=M_target,
            coeff_res_rel=coeff_res_rel,
            freq_res_rel=freq_res_rel,
            imag_res=imag_res,
            min_P=min_P,
            halfband_residual=hb_err
        )

        diag = FactorizationDiagnostics(
            mode=mode,
            target_degree=M_target,
            actual_degree=actual_deg,
            coefficient_residual_abs=coeff_res_abs,
            coefficient_residual_rel=coeff_res_rel,
            frequency_residual_abs=freq_res_abs,
            frequency_residual_rel=freq_res_rel,
            factor_imaginary_residual=imag_res,
            positivity_min=min_P,
            positivity_max=max_P,
            halfband_residual=hb_err,
            zero_on_unit_circle_count=zero_uc_count,
            finite=is_finite,
            certified=certified
        )
        return h0, diag

    @staticmethod
    def certify_factorization_diagnostics(
        is_finite: bool,
        actual_degree: int,
        target_degree: int,
        coeff_res_rel: float,
        freq_res_rel: float,
        imag_res: float,
        min_P: float,
        halfband_residual: float = 0.0
    ) -> bool:
        """Single-source authority for certifying spectral factor diagnostic payloads."""
        return bool(
            is_finite and
            actual_degree == target_degree and
            coeff_res_rel <= 1e-2 and
            freq_res_rel <= 1e-2 and
            imag_res <= 1e-5 and
            min_P >= -1e-3 and
            halfband_residual <= 1e-3
        )

    def spectral_factor_cepstral(self, p_taps: np.ndarray, target_N: int) -> Tuple[np.ndarray, float]:
        """
        Constructs an FFT/cepstrum approximation to the minimum-phase spectral factor and truncates it to target_N taps.
        Operates on positive trigonometric power polynomial P(w) >= 0.
        Returns (h0_taps, residual).
        """
        M = target_N - 1
        center = (len(p_taps) - 1) // 2
        K_grid = 16384

        # Evaluate trigonometric polynomial P(w) = sum p_k e^{-j k w}
        w = 2.0 * np.pi * np.arange(K_grid) / float(K_grid)
        P_w = np.zeros(K_grid, dtype=np.float64)
        for k_idx, val in enumerate(p_taps):
            P_w += val * np.cos((k_idx - center) * w)

        P_w = np.maximum(P_w, 1e-14)
        log_P = 0.5 * np.log(P_w)

        # Real cepstrum via IFFT
        c = np.fft.ifft(log_P).real

        # Construct minimum-phase causal log-spectrum
        c_min = np.zeros(K_grid, dtype=np.float64)
        c_min[0] = c[0]
        c_min[1:K_grid // 2] = 2.0 * c[1:K_grid // 2]
        c_min[K_grid // 2] = c[K_grid // 2]

        # Minimum-phase spectral response H_min(w) = exp(FFT(c_min))
        H_min = np.exp(np.fft.fft(c_min))
        h_full = np.fft.ifft(H_min).real

        # Truncate to target N taps
        h0 = h_full[:target_N]

        # Scale h0 via optimal least-squares scalar fit
        h0 = _fit_spectral_factor_scale(h0, p_taps)

        # Autocorrelation residual
        r_fact = np.convolve(h0, h0[::-1])
        res = float(np.max(np.abs(r_fact - p_taps)))
        return h0, res

    def factor_by_chebyshev_roots(self, p_taps: np.ndarray, target_N: int) -> Tuple[np.ndarray, float]:
        """
        Structured Chebyshev x-domain root factorization with unit-circle conjugate pairing.
        Converts P(z) = b0 + sum_{m=1}^M b_m T_m(x) into degree-M polynomial in x = cos(w).
        Finds degree-M roots in x and lifts z_j = x_j - sqrt(x_j^2 - 1) to enforce exact reciprocal symmetry.
        Symmetrically pairs unit-circle roots x in [-1, 1] as e^{j theta} and e^{-j theta} to preserve real FIR taps.
        Returns (h0_taps, residual).
        """
        M = target_N - 1
        center = (len(p_taps) - 1) // 2

        # Extract Fourier cosine coefficients b_m
        b_harmonics = np.zeros(M + 1, dtype=np.float64)
        b_harmonics[0] = p_taps[center]
        for m in range(1, M + 1):
            if center + m < len(p_taps):
                b_harmonics[m] = 2.0 * p_taps[center + m]

        # Chebyshev roots in x = cos(w)
        x_roots = np.polynomial.chebyshev.chebroots(b_harmonics)

        # Lift x_roots to z_roots in unit circle |z| <= 1 with conjugate pairing for interior x roots
        z_inside = []
        unit_x = [x for x in x_roots if abs(np.imag(x)) < 1e-6 and -1.0 - 1e-4 <= x.real <= 1.0 + 1e-4]
        off_x = [x for x in x_roots if x not in unit_x]

        # Process off-unit-circle roots
        for x_val in off_x:
            disc = np.sqrt(x_val**2 - 1.0 + 0j)
            z1 = x_val - disc
            z2 = x_val + disc
            z_sel = z1 if abs(z1) <= 1.0 + 1e-4 else z2
            z_inside.append(z_sel)

        # Process unit-circle roots x in [-1, 1] as conjugate pairs e^{+-j theta}
        unit_reals = sorted([np.clip(x.real, -1.0, 1.0) for x in unit_x])
        u_idx = 0
        while u_idx < len(unit_reals):
            xr = unit_reals[u_idx]
            if u_idx + 1 < len(unit_reals) and abs(unit_reals[u_idx + 1] - xr) < 1e-3:
                # Pair double root as e^{+j theta} and e^{-j theta} using centroid for stability
                xr_avg = 0.5 * (xr + unit_reals[u_idx + 1])
                theta = np.arccos(xr_avg)
                z_inside.append(np.exp(1j * theta))
                z_inside.append(np.exp(-1j * theta))
                u_idx += 2
            else:
                disc = np.sqrt(xr**2 - 1.0 + 0j)
                z_inside.append(xr - disc)
                u_idx += 1

        h0_complex = np.poly(z_inside)
        if len(h0_complex) < target_N:
            h0_complex = np.pad(h0_complex, (0, target_N - len(h0_complex)), mode='constant')
        elif len(h0_complex) > target_N:
            raise ValueError(f"Structured factor degree mismatch: got {len(h0_complex) - 1}, expected {target_N - 1}.")

        h0_complex = _fit_spectral_factor_scale(h0_complex, p_taps)

        res = float(np.max(np.abs(np.convolve(np.real(h0_complex), np.real(h0_complex)[::-1]) - p_taps)))
        return h0_complex, res

    def factor_by_roots_reference(self, p_taps: np.ndarray, target_N: int) -> Tuple[np.ndarray, float]:
        """
        Reference degree-2M monomial root finding backend (np.roots).
        """
        M = target_N - 1
        roots = np.roots(p_taps)

        rel_tol, abs_tol = 1e-2, 1e-3
        def same_root(a, b):
            return abs(a - b) <= max(abs_tol, rel_tol * max(abs(a), abs(b)))

        orbits = []
        used = [False] * len(roots)
        for i in range(len(roots)):
            if used[i]: continue
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

        inside_roots = []
        for orbit in orbits:
            in_orbit = [r for r in orbit if abs(r) <= 1.0 + 1e-4]
            half_len = len(orbit) // 2
            if len(in_orbit) == half_len or len(in_orbit) == len(orbit):
                inside_roots.extend(in_orbit)
            elif half_len > 0:
                # Decompose reciprocal/conjugate orbit strictly
                sorted_orb = sorted(orbit, key=lambda x: abs(x))
                inside_roots.extend(sorted_orb[:half_len])
            else:
                raise ValueError(f"Structural orbit decomposition failure for root orbit: {orbit}")

        if len(inside_roots) != M:
            if len(inside_roots) > M:
                inside_roots = sorted(inside_roots, key=lambda x: abs(x))[:M]
            else:
                raise ValueError(f"Reference root factorization degree mismatch: got {len(inside_roots)}, expected {M}.")

        h0_complex = np.poly(inside_roots)
        h0_complex = _fit_spectral_factor_scale(h0_complex, p_taps)

        res = float(np.max(np.abs(np.convolve(np.real(h0_complex), np.real(h0_complex)[::-1]) - p_taps)))
        return h0_complex, res

    def spectral_factor_power_polynomial(
        self,
        p_taps: np.ndarray,
        target_N: int,
        mode: FactorMode = FactorMode.STRUCTURED_CHEBYSHEV
    ) -> Tuple[np.ndarray, FactorizationDiagnostics]:
        """
        Spectrally factors positive half-band power polynomial P(z) into minimum-phase factor H0(z) of length target_N.
        Dispatches to Structured Chebyshev x-domain root lifting, Cepstral approximation, or Reference Root backends.
        Returns (h0_taps, diagnostics).
        """
        if mode == FactorMode.STRUCTURED_CHEBYSHEV:
            h0_raw, _ = self.factor_by_chebyshev_roots(p_taps, target_N)
        elif mode == FactorMode.CEPSTRAL_APPROX:
            h0_raw, _ = self.spectral_factor_cepstral(p_taps, target_N)
        elif mode == FactorMode.REFERENCE_ROOTS:
            h0_raw, _ = self.factor_by_roots_reference(p_taps, target_N)
        else:
            raise ValueError(f"Unknown FactorMode: '{mode}'")

        h0, diag = self._compute_factorization_diagnostics(
            h0_raw, p_taps, mode=mode, target_degree=target_N - 1
        )
        return h0, diag

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
                    phi_k = self._eval_basis(k, nodes)
                    c1 = float(c_n_1_val(k, self.lam))
                    norm_sq = phi_norm_squared(k, self.lam) / (c1 ** 2) if self.basis_type == "normalized" else phi_norm_squared(k, self.lam)
                    a_coeffs[k] = np.sum(D_q * phi_k * weights) / norm_sq
                A = np.zeros((self.grid_samples, K))
                for k in range(K):
                    A[:, k] = self._eval_basis(k, nodes)
                sqrt_w = np.sqrt(weights)
                A_w = A * sqrt_w[:, np.newaxis]
                s_vals = np.linalg.svd(A_w, compute_uv=False)
                cond_val = float(s_vals[0] / s_vals[-1]) if len(s_vals) > 0 and s_vals[-1] > 1e-12 else np.inf
                res_data = float(np.linalg.norm(sqrt_w * (A @ a_coeffs - D_q)) / max(np.linalg.norm(sqrt_w * D_q), 1e-15))
                return a_coeffs, K, cond_val, res_data, res_data, 0.0
            else:
                # Direct QR/SVD least squares solve using FIR symmetry envelope basis
                A = np.zeros((self.grid_samples, K))
                for k in range(K):
                    A[:, k] = self._eval_fir_basis(k, nodes, symmetry=sym)
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
                A[:, k] = self._eval_fir_basis(k, nodes, symmetry=sym)

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
                A[:, k] = self._eval_fir_basis(k, x, symmetry=sym)

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
            A_freq += c * self._eval_fir_basis(k, x, symmetry=sym)

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
            else:
                K_fft_norm = max(4096, 1 << (math.ceil(math.log2(spec.order)) + 2))
                max_g = np.max(np.abs(np.fft.fft(h, K_fft_norm)))
                if max_g > 1e-12:
                    h /= max_g
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
            f"#define GEG_E_TOTAL_ESTIMATE {p.total}f",
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

        factor_diag = None
        factorization_certified = True

        if spec.kind in ("qmf", "asymmetric_qmf"):
            # Direct QMF compilation route via half-band power polynomial and spectral factorization
            a_coeffs, p_taps, K, cond_val, augmented_design_residual, data_design_residual, regularization_residual = self.solve_halfband_power_polynomial(spec)
            power_valid, min_P, max_P, hb_err = self.validate_power_polynomial(p_taps)
            if not power_valid:
                raise ValueError(
                    f"Invalid half-band power polynomial: min P = {min_P:.3e}, max P = {max_P:.3e}, "
                    f"halfband_error = {hb_err:.3e} exceeds positivity/complementarity tolerance."
                )

            h0_float, factor_diag = self.spectral_factor_power_polynomial(p_taps, target_N=spec.order, mode=self.factor_mode)
            factorization_certified = factor_diag.certified
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
            a_coeffs, K, cond_val, augmented_design_residual, data_design_residual, regularization_residual = self.solve_coefficients(spec)
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

        if spec.kind in ("qmf", "asymmetric_qmf"):
            basis_degrees = 2 * np.arange(K) + 1
            a_active = a_coeffs[basis_degrees]
            op_eigs = basis_degrees * (basis_degrees + 2.0 * self.lam)
            reg_energy = float(np.sum((a_active ** 2) * (op_eigs ** self.reg_power)))
        else:
            basis_degrees = np.arange(K)
            op_eigs = basis_degrees * (basis_degrees + 2.0 * self.lam)
            reg_energy = float(np.sum((a_coeffs[:K] ** 2) * (op_eigs ** self.reg_power)))

        asymp_err = 0.0
        matching_status = MatchingStatus.MATCHING_SCHEMA
        if self.asymptotic_mode != "none":
            omega_sample = np.linspace(0.001, np.pi - 0.001, 100)
            x_sample = np.cos(omega_sample)

            # Compute asymptotic response error bound over active degrees
            deg_errs = []
            weighted_response_bound = 0.0
            for deg_idx, deg in enumerate(basis_degrees):
                deg_int = int(deg)
                if deg_int == 0:
                    continue
                phi_exact = normalized_phi_recurrence(deg_int, self.lam, x_sample)
                if self.basis_type == "unnormalized":
                    phi_exact = phi_exact * float(c_n_1_val(deg_int, self.lam))
                phi_asymp = self._eval_pure_asymptotic_basis(deg_int, omega_sample)
                max_eps_n = float(np.max(np.abs(phi_exact - phi_asymp)))
                deg_errs.append(max_eps_n)
                coeff_mag = float(np.abs(a_coeffs[deg_int])) if deg_int < len(a_coeffs) else 0.0
                weighted_response_bound += coeff_mag * max_eps_n

            asymp_err = float(np.max(deg_errs)) if len(deg_errs) > 0 else 0.0

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

        # Layer VII & VIII tri-state verification status evaluation
        if self.asymptotic_mode == "none":
            basis_asymptotic = VerificationStatus.NOT_APPLICABLE
            asymptotic_diagnostic_passed = True
        else:
            asymp_ok = bool(asymp_err < 0.05)
            basis_asymptotic = VerificationStatus.VERIFIED if asymp_ok else VerificationStatus.FAILED
            asymptotic_diagnostic_passed = asymp_ok

        # Exact design target compliance
        exact_pass_target = spec.passband_ripple_db
        exact_stop_target = spec.stopband_atten_db

        design_target_met = bool(
            pass_ripple <= exact_pass_target and
            stop_atten >= exact_stop_target
        )
        if spec.kind in ("qmf", "asymmetric_qmf") and h1_quant is not None:
            design_target_met = design_target_met and bool(
                pass_ripple_h1 <= exact_pass_target and
                stop_atten_h1 >= exact_stop_target
            )

        # Relaxed engineering acceptance limits
        relaxed_pass_limit = exact_pass_target * 3.0 if spec.kind in ("qmf", "asymmetric_qmf") else exact_pass_target * 2.0
        relaxed_stop_limit = min(exact_stop_target * 0.5, 15.0)

        engineering_acceptance = bool(
            pass_ripple <= relaxed_pass_limit and
            stop_atten >= relaxed_stop_limit
        )
        if spec.kind in ("qmf", "asymmetric_qmf") and h1_quant is not None:
            engineering_acceptance = engineering_acceptance and bool(
                pass_ripple_h1 <= relaxed_pass_limit and
                stop_atten_h1 >= relaxed_stop_limit
            )

        prototype_fir = VerificationStatus.VERIFIED if engineering_acceptance else VerificationStatus.FAILED
        specification_met = design_target_met

        # Realization properties verification (finite coefficients, non-empty taps)
        realization_verified = bool(
            np.all(np.isfinite(h0_quant.float64_taps)) and
            len(h0_quant.float64_taps) == spec.order
        )

        if spec.kind in ("qmf", "asymmetric_qmf"):
            factor_status = VerificationStatus.VERIFIED if factorization_certified else VerificationStatus.FAILED
            qmf_pow_ok = bool(qmf_pow_lin <= 0.05)
            qmf_alias_ok = bool(qmf_alias_lin <= 0.05)
            qmf_power = VerificationStatus.VERIFIED if qmf_pow_ok else VerificationStatus.FAILED
            qmf_alias = VerificationStatus.VERIFIED if qmf_alias_ok else VerificationStatus.FAILED

            structural_certified = (
                realization_verified and
                factor_status == VerificationStatus.VERIFIED and
                qmf_power == VerificationStatus.VERIFIED and
                qmf_alias == VerificationStatus.VERIFIED
            )
        else:
            factor_status = VerificationStatus.NOT_APPLICABLE
            qmf_power = VerificationStatus.NOT_APPLICABLE
            qmf_alias = VerificationStatus.NOT_APPLICABLE

            # For ordinary FIR, structural certification is given by realization verification
            structural_certified = realization_verified

        is_certified = structural_certified and specification_met

        payload = CertifiedEvaluationPayload(
            truth_status=truth_status,
            matching_status=matching_status,
            provenance=provenance,
            basis_asymptotic=basis_asymptotic,
            prototype_fir=prototype_fir,
            qmf_power=qmf_power,
            qmf_alias=qmf_alias,
            factorization=factor_status,
            structural_certified=structural_certified,
            specification_met=specification_met,
            asymptotic_diagnostic_passed=asymptotic_diagnostic_passed,
            is_certified=is_certified
        )

        # Build comprehensive EpistemicContract with explicit claim warrants
        epistemic_contract = EpistemicContract(
            target_name=f"FilterResult_{spec.kind}_{spec.order}",
            target_type="FIR_FILTER",
            required_claims=["REALIZATION_VERIFIED", "DESIGN_SPECIFICATION_COMPLIANCE"]
        )

        epistemic_contract.add_warrant(ClaimWarrant(
            proposition="REALIZATION_VERIFIED",
            target_type="FIR_FILTER",
            domain="TAPS_ARRAY",
            disposition=WarrantDisposition.ESTABLISHED if realization_verified else WarrantDisposition.CONTRADICTED,
            method=EvidenceMethod.EXACT_ARITHMETIC,
            evidence_payload={"order": spec.order, "finite_taps": bool(np.all(np.isfinite(h0_quant.float64_taps)))}
        ))

        epistemic_contract.add_warrant(ClaimWarrant(
            proposition="DESIGN_SPECIFICATION_COMPLIANCE",
            target_type="DSP_SPECIFICATION",
            domain="FREQUENCY_GRID",
            disposition=WarrantDisposition.ESTABLISHED if design_target_met else WarrantDisposition.CONTRADICTED,
            method=EvidenceMethod.ORDINARY_NUMERICAL_DIAGNOSTIC,
            evidence_payload={"pass_ripple": float(pass_ripple), "stop_atten": float(stop_atten)}
        ))

        if spec.kind in ("qmf", "asymmetric_qmf"):
            epistemic_contract.required_claims.extend(["POWER_COMPLEMENTARITY", "ALIAS_CANCELLATION"])
            epistemic_contract.add_warrant(ClaimWarrant(
                proposition="POWER_COMPLEMENTARITY",
                target_type="HALF_BAND_POWER",
                domain="FREQUENCY_GRID",
                disposition=WarrantDisposition.ESTABLISHED if qmf_pow_ok else WarrantDisposition.CONTRADICTED,
                method=EvidenceMethod.VALIDATED_NUMERICAL_BOUND,
                evidence_payload={"qmf_power_error_linear": qmf_pow_lin}
            ))
            epistemic_contract.add_warrant(ClaimWarrant(
                proposition="ALIAS_CANCELLATION",
                target_type="CQF_BANK",
                domain="FREQUENCY_GRID",
                disposition=WarrantDisposition.ESTABLISHED if qmf_alias_ok else WarrantDisposition.CONTRADICTED,
                method=EvidenceMethod.VALIDATED_NUMERICAL_BOUND,
                evidence_payload={"qmf_alias_error_linear": qmf_alias_lin}
            ))

        # Recompute final realization frequency fit residual from final normalized taps
        H0_final = np.abs(np.fft.fft(h0_quant.float64_taps, K_fft)[:K_fft // 2 + 1])
        omega_eval = 2.0 * np.pi * freq_grid
        D_eval, W_eval = self._build_spectral_target(spec, omega_eval)
        final_realization_residual = float(np.linalg.norm(np.sqrt(W_eval) * (H0_final - D_eval)) / max(np.linalg.norm(np.sqrt(W_eval) * D_eval), 1e-15))

        prov_err = float(np.max(mixed_error(h0_quant.float64_taps, h0_quant.q15_taps / h0_quant.q15_scale)))

        spectral_design = GegenbauerSpectralDesign(
            lam=self.lam,
            a_coeffs=a_coeffs,
            basis_degrees=basis_degrees,
            basis_terms=K,
            operator_eigenvalues=op_eigs,
            sturm_liouville_energy=reg_energy,
            projection_residual=final_realization_residual,
            quadrature_residual=augmented_design_residual,
            conditioning=cond_val,
            asymptotic_sampled_discrepancy=asymp_err,
            truth_status=truth_status,
            matching_status=matching_status,
            augmented_design_residual=augmented_design_residual,
            data_design_residual=data_design_residual,
            regularization_residual=regularization_residual,
            final_realization_residual=final_realization_residual
        )

        result = FilterResult(
            spec=spec,
            lam=self.lam,
            basis_terms=K,
            mu_reg=self.mu_reg,
            h0_taps=h0_quant,
            payload=payload,
            epistemic_contract=epistemic_contract,
            factorization=factor_diag,
            h1_taps=h1_quant,
            spectral_design=spectral_design,
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
            fit_residual=augmented_design_residual,
            data_fit_residual=final_realization_residual,
            regularization_residual=regularization_residual,
            augmented_design_residual=augmented_design_residual,
            data_design_residual=data_design_residual,
            final_realization_residual=final_realization_residual,
            asymptotic_sampled_discrepancy=asymp_err,
            provenance_mixed_error=prov_err
        )
        result.header_code = self.generate_header(result)
        return result

    def compile_biorthogonal_pair(self, order_h0: int, order_g0: int, cutoff: float = 0.25) -> dict:
        """
        Compiles a Biorthogonal filter bank pair (H0, G0) of polynomial degrees order_h0 and order_g0.
        Uses Gegenbauer half-band product filter factorization.
        The product filter E(z) = H0(z) G0(z) = 2 z^-d P_c(z) where P_c(z) is the centered half-band
        Laurent polynomial satisfying P_c(z) + P_c(-z) = 1.
        For odd delay d = (order_h0 + order_g0) // 2, E(z) - E(-z) = 2 z^-d (P_c(z) + P_c(-z)) = 2 z^-d (exact PR delay).
        Requires odd PR delay (order_h0 + order_g0 == 2 mod 4) for standard 2-channel linear-phase half-band PR.
        """
        total_order = order_h0 + order_g0
        if total_order % 2 != 0:
            raise ValueError("The sum of H0 and G0 orders must be even for a valid half-band filter.")

        delay = total_order // 2
        if delay % 2 == 0:
            raise ValueError(
                f"Current 2-channel half-band biorthogonal construction requires odd PR delay (delay = {delay} is even). "
                f"Use degree pairs satisfying order_h0 + order_g0 == 2 mod 4 (e.g. 4/2, 8/6, 12/10)."
            )

        trans_half = min(0.24, max(0.01, abs(cutoff - 0.25) if abs(cutoff - 0.25) > 1e-4 else 0.05))
        p_spec = FilterSpec(
            kind="qmf",
            order=(total_order // 2) + 1,
            cutoff=0.25,
            wp=0.25 - trans_half,
            ws=0.25 + trans_half
        )

        _, p_taps, _, _, _, _, _ = self.solve_halfband_power_polynomial(p_spec)

        # Set center tap to Bootstrapped half-band value
        center = total_order // 2
        p_taps[center] = 0.5
        p_prod = 2.0 * p_taps

        # Rescale off-center taps so P_prod(1) == 2.0 exactly while preserving P_prod(center) = 1.0
        off_center_sum = float(np.sum(p_prod) - p_prod[center])
        if abs(off_center_sum) > 1e-12:
            p_prod[:center] *= (1.0 / off_center_sum)
            p_prod[center + 1:] *= (1.0 / off_center_sum)

        # Verify product polynomial DC gain before allocating roots
        p_prod_dc = float(np.sum(p_prod))
        if abs(p_prod_dc - 2.0) > 1e-3:
            raise ValueError(f"Product polynomial DC gain {p_prod_dc:.6f} deviates from target 2.0 (requires H0(1)G0(1) = 2.0).")

        # Build atomic root clusters via Chebyshev x-domain roots in x = cos(w)
        M_prod = total_order
        b_harmonics = np.zeros(M_prod + 1, dtype=np.float64)
        b_harmonics[0] = p_prod[center]
        for m in range(1, M_prod + 1):
            if center + m < len(p_prod):
                b_harmonics[m] = 2.0 * p_prod[center + m]

        x_roots = np.polynomial.chebyshev.chebroots(b_harmonics)
        roots = []
        for x_val in x_roots:
            disc = np.sqrt(x_val**2 - 1.0 + 0j)
            roots.extend([x_val - disc, x_val + disc])
        rel_tol, abs_tol = 1e-2, 1e-3
        def same_root(a, b):
            return abs(a - b) <= max(abs_tol, rel_tol * max(abs(a), abs(b)))

        raw_clusters = []
        used_r = [False] * len(roots)
        for i in range(len(roots)):
            if used_r[i]:
                continue
            group = [i]
            used_r[i] = True
            for j in range(i + 1, len(roots)):
                if not used_r[j] and same_root(roots[j], roots[i]):
                    group.append(j)
                    used_r[j] = True
            rep = np.mean([roots[k] for k in group])
            raw_clusters.append({'rep': rep, 'mult': len(group)})

        # Group raw clusters into symmetry units (reciprocal & conjugate quadruplets / pairs & real endpoints)
        used_c = [False] * len(raw_clusters)
        atomic_units = []
        for i in range(len(raw_clusters)):
            if used_c[i]:
                continue
            c1 = raw_clusters[i]
            r1 = c1['rep']
            m1 = c1['mult']
            used_c[i] = True

            if abs(np.imag(r1)) < 1e-3:
                r1_real = float(np.real(r1))
                if abs(abs(r1_real) - 1.0) < 1e-3:
                    # Endpoint root at z = +1 or z = -1 (splittable single real root)
                    ep_val = 1.0 if r1_real > 0 else -1.0
                    atomic_units.append({'kind': 'REAL_ENDPOINT', 'roots': [ep_val], 'size': 1, 'mult': m1})
                else:
                    recip_idx = -1
                    for j in range(len(raw_clusters)):
                        if not used_c[j] and abs(raw_clusters[j]['rep'] - 1.0/r1_real) < 1e-2:
                            recip_idx = j
                            break
                    if recip_idx != -1:
                        c2 = raw_clusters[recip_idx]
                        if m1 != c2['mult']:
                            raise ValueError(f"Reciprocal orbit multiplicities disagree for root {r1_real:.6f}: {m1} vs {c2['mult']}.")
                        used_c[recip_idx] = True
                        atomic_units.append({'kind': 'REAL_PAIR', 'roots': [r1_real, float(np.real(c2['rep']))], 'size': 2, 'mult': m1})
                    else:
                        raise ValueError(f"Unpaired real reciprocal root detected: {r1_real:.6f} without reciprocal partner {1.0/r1_real:.6f}.")
            else:
                quad_indices = [i]
                for j in range(len(raw_clusters)):
                    if not used_c[j]:
                        r2 = raw_clusters[j]['rep']
                        if abs(r2 - np.conj(r1)) < 1e-2 or abs(r2 - 1.0/r1) < 1e-2 or abs(r2 - 1.0/np.conj(r1)) < 1e-2:
                            quad_indices.append(j)
                            used_c[j] = True
                quad_roots = [raw_clusters[k]['rep'] for k in quad_indices]
                multiplicities = [raw_clusters[k]['mult'] for k in quad_indices]
                if len(set(multiplicities)) != 1:
                    raise ValueError(f"Complex root orbit multiplicities disagree: {multiplicities}")
                quad_mult = multiplicities[0]
                kind = 'UNIT_PAIR' if abs(abs(r1) - 1.0) < 1e-3 else 'COMPLEX_QUARTET'
                expected_size = 2 if kind == 'UNIT_PAIR' else 4
                if len(quad_roots) != expected_size:
                    raise ValueError(f"Invalid root orbit size for {kind}: got {len(quad_roots)}, expected {expected_size}.")
                atomic_units.append({'kind': kind, 'roots': quad_roots, 'size': len(quad_roots), 'mult': quad_mult})

        # Verify exact algebraic orbit closure: sum size_i * mult_i == total_order
        orbit_total_size = sum(unit['size'] * unit['mult'] for unit in atomic_units)
        if orbit_total_size != total_order:
            raise ValueError(
                f"Root-orbit decomposition is not closed: got {orbit_total_size} roots, expected {total_order}."
            )

        # Generate candidate root allocations for target order_h0
        allocations = []
        def search_allocations(unit_idx, current_deg_h0, current_h0_roots, current_g0_roots):
            if unit_idx == len(atomic_units):
                if current_deg_h0 == order_h0:
                    allocations.append((current_h0_roots, current_g0_roots))
                return

            unit = atomic_units[unit_idx]
            kind, r_list, size, mult = unit['kind'], unit['roots'], unit['size'], unit['mult']

            if kind == 'REAL_ENDPOINT':
                # Endpoint roots (z = +-1) can be partitioned in steps of 1
                for k in range(0, mult + 1):
                    if current_deg_h0 + k * size <= order_h0:
                        search_allocations(
                            unit_idx + 1,
                            current_deg_h0 + k * size,
                            current_h0_roots + r_list * k,
                            current_g0_roots + r_list * (mult - k)
                        )
            else:
                # Pair / Quartet multiplicity allocation
                for k in range(0, mult + 1):
                    if current_deg_h0 + k * size <= order_h0:
                        search_allocations(
                            unit_idx + 1,
                            current_deg_h0 + k * size,
                            current_h0_roots + r_list * k,
                            current_g0_roots + r_list * (mult - k)
                        )

        # Use degree-bounded branch & bound pruning search
        search_allocations(0, 0, [], [])

        if not allocations:
            raise ValueError(
                f"Unable to partition roots into exact target orders order_h0={order_h0} and order_g0={order_g0} "
                f"while preserving symmetric root clusters."
            )

        # Rank candidate root factorizations by DSP lowpass performance
        best_candidate = None
        best_score = float('inf')

        imag_tol = 1e-5
        dc_tol = 1e-6

        for cand_h0_roots, cand_g0_roots in allocations:
            h0_u_complex = np.poly(cand_h0_roots) if len(cand_h0_roots) > 0 else np.array([1.0 + 0j])
            g0_u_complex = np.poly(cand_g0_roots) if len(cand_g0_roots) > 0 else np.array([1.0 + 0j])

            h0_imag = float(np.max(np.abs(np.imag(h0_u_complex))))
            g0_imag = float(np.max(np.abs(np.imag(g0_u_complex))))
            if max(h0_imag, g0_imag) > imag_tol:
                continue

            h0_u = np.real(h0_u_complex)
            g0_u = np.real(g0_u_complex)

            conv_u = np.convolve(h0_u, g0_u)
            c = float(p_prod[0] / conv_u[0]) if abs(conv_u[0]) > 1e-12 else 1.0
            h0_c = h0_u * np.sign(c) * np.sqrt(abs(c))
            g0_c = g0_u * np.sqrt(abs(c))

            # Pure reciprocal scale normalization preserving exact H0(z) G0(z) = P_prod(z) product polynomial
            h0_sum = float(np.sum(h0_c))
            if abs(h0_sum) <= dc_tol:
                continue

            scale_h0 = np.sqrt(2.0) / h0_sum
            h0_norm = h0_c * scale_h0
            g0_norm = g0_c / scale_h0

            sym_h0 = np.max(np.abs(h0_norm - h0_norm[::-1]))
            sym_g0 = np.max(np.abs(g0_norm - g0_norm[::-1]))

            # Enforce symmetry structurally as a hard constraint
            if max(sym_h0, sym_g0) > 1e-4:
                continue

            dc_err_h0 = abs(np.sum(h0_norm) - np.sqrt(2.0))
            dc_err_g0 = abs(np.sum(g0_norm) - np.sqrt(2.0))

            nyq_h0 = abs(np.sum(h0_norm * np.array([(-1.0)**n for n in range(len(h0_norm))])))
            nyq_g0 = abs(np.sum(g0_norm * np.array([(-1.0)**n for n in range(len(g0_norm))])))

            # Evaluate response metrics over frequency grid
            K_eval = 512
            H0_eval = np.abs(np.fft.fft(h0_norm, K_eval)[:K_eval // 2 + 1])
            G0_eval = np.abs(np.fft.fft(g0_norm, K_eval)[:K_eval // 2 + 1])
            f_grid = np.arange(K_eval // 2 + 1) / float(K_eval)

            p_idx = f_grid <= cutoff * 0.8
            s_idx = f_grid >= cutoff * 1.2

            ripple_h0 = (np.max(H0_eval[p_idx]) - np.min(H0_eval[p_idx])) if np.any(p_idx) else 0.0
            ripple_g0 = (np.max(G0_eval[p_idx]) - np.min(G0_eval[p_idx])) if np.any(p_idx) else 0.0

            stop_h0 = np.max(H0_eval[s_idx]) if np.any(s_idx) else 0.0
            stop_g0 = np.max(G0_eval[s_idx]) if np.any(s_idx) else 0.0

            score = (
                (dc_err_h0 + dc_err_g0) * 50.0 +
                (nyq_h0 + nyq_g0) * 20.0 +
                (ripple_h0 + ripple_g0) * 10.0 +
                (stop_h0 + stop_g0) * 10.0 +
                0.1 * (np.max(np.abs(h0_norm)) + np.max(np.abs(g0_norm)))
            )

            if score < best_score:
                best_score = score
                best_candidate = (h0_norm, g0_norm)

        if best_candidate is None:
            raise ValueError("No valid biorthogonal lowpass candidate satisfied symmetry and DC constraints.")

        h0_taps, g0_taps = best_candidate

        # Hard postcondition check for individual lowpass factor DC gains H0(1) == sqrt(2) and G0(1) == sqrt(2)
        h0_dc_val = float(np.sum(h0_taps))
        g0_dc_val = float(np.sum(g0_taps))
        target_dc = np.sqrt(2.0)
        if abs(h0_dc_val - target_dc) > 1e-3 or abs(g0_dc_val - target_dc) > 1e-3:
            raise ValueError(
                f"Biorthogonal factor DC gain postcondition failed: H0(1)={h0_dc_val:.6f}, G0(1)={g0_dc_val:.6f}, expected sqrt(2)={target_dc:.6f}."
            )

        # Generate complementary highpass filters H1 and G1 for 2-channel Biorthogonal Bank
        # H1(z) = G0(-z) => h1[n] = (-1)^n * g0[n]
        # G1(z) = -H0(-z) => g1[n] = -(-1)^n * h0[n]
        # This yields H0(z)G0(z) + H1(z)G1(z) = P(z) + P(-z) = 2 z^-delay (Perfect Reconstruction for odd delay)
        # and H0(-z)G0(z) + H1(-z)G1(z) = H0(-z)G0(z) - G0(z)H0(-z) = 0 (Exact Alias Cancellation)
        h1_taps = np.array([((-1.0)**n) * g0_taps[n] for n in range(len(g0_taps))])
        g1_taps = np.array([-((-1.0)**n) * h0_taps[n] for n in range(len(h0_taps))])

        # Verification metrics & symmetry residuals (comparing H0 * G0 against P_prod(z))
        h0_conv_g0 = np.convolve(h0_taps, g0_taps)
        product_residual = float(np.max(np.abs(h0_conv_g0 - p_prod)))
        if product_residual > 1e-6:
            raise ValueError(f"Biorthogonal polynomial factorization residual {product_residual:.4e} exceeds 1e-6 tolerance.")

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
        if pr_residual > 1e-5:
            raise ValueError(f"Biorthogonal PR response residual {pr_residual:.4e} exceeds 1e-5 tolerance.")

        # Alias cancellation check H0(-z) G0(z) + H1(-z) G1(z) == 0
        H0_neg = np.fft.fft(h0_taps * np.array([(-1.0)**n for n in range(len(h0_taps))]), K_fft)
        H1_neg = np.fft.fft(h1_taps * np.array([(-1.0)**n for n in range(len(h1_taps))]), K_fft)
        alias_complex = H0_neg * G0_f + H1_neg * G1_f
        alias_residual = float(np.max(np.abs(alias_complex)))
        if alias_residual > 1e-5:
            raise ValueError(f"Biorthogonal alias cancellation residual {alias_residual:.4e} exceeds 1e-5 tolerance.")

        # Structural frequency domain relations H1(w) = G0(w+pi) and G1(w) = -H0(w+pi)
        G0_shift = np.roll(G0_f, K_fft // 2)
        H0_shift = np.roll(H0_f, K_fft // 2)
        h1_qmf_res = float(np.max(np.abs(H1_f - G0_shift)))
        g1_qmf_res = float(np.max(np.abs(G1_f + H0_shift)))

        return {
            "H0": self.quantize_taps(h0_taps),
            "H1": self.quantize_taps(h1_taps),
            "G0": self.quantize_taps(g0_taps),
            "G1": self.quantize_taps(g1_taps),
            "P": p_prod * 0.5,
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
            alias_transfer = qmf_alias_transfer(H0, H1)
            pow_db = 10 * np.log10(np.maximum(1e-12, pow_comp[:len(result.freq_grid)]))
            alias_db = 20 * np.log10(np.maximum(1e-12, np.abs(alias_transfer[:len(result.freq_grid)])))

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
        lambda_candidates: List[float] = [0.5, 1.0, 1.25, 1.5],
        mu_candidates: List[float] = [0.0, 1e-4],
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
                    factor_mode=self.factor_mode,
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
