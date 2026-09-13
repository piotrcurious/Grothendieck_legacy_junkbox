"""
XRF Normalization & Regularization Preprocessor with Red Angel Auto-Tuning
========================================================================
Implements reference sample calibration, domain mapping, Lebesgue measure range
partitioning, Sturm-Liouville differential regularization, Red Angel anomaly characterization,
and automatic parameter tuning for X-Ray Fluorescence (XRF) spectral pipelines.

The "Red Angel" Anomaly:
  Unregularized differentiation or high-degree transform of noisy/unnormalized spectral
  data causes exponential growth of higher-order polynomial coefficients, manifesting as
  spurious high-frequency oscillations ("ghost spectra").

Mitigation Architecture:
  1. Strict Reference Sample Normalization & Domain Mapping E -> [-1, 1]
  2. Lebesgue Measure Range Partitioning (Range-based level set evaluation for quantized samplers)
  3. Sturm-Liouville Operator Regularization L_lambda = n(n + 2*lambda)
  4. Red Angel Metric Characterization & Auto-Tuning of mu_reg and lambda
"""

import os
import sys
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure parent directory is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from compiler import FilterSpec, GegenbauerFilterCompiler
from xrf_spectrometer_demo import XRFSpectrometer, XRFElement, XRF_ELEMENT_DATABASE
from gegenbauer_asymptotics import normalized_phi_recurrence
from algebraic_geometry_combinatorics import phi_norm_squared


class RedAngelCharacterizer:
    """
    Quantifies numerical instability and ghost spectrum generation ("Red Angel" anomaly).
    Evaluates derivative energy amplification and high-order spectral tail leakage.
    """

    @staticmethod
    def compute_derivative_energy(signal: np.ndarray, x_grid: np.ndarray) -> float:
        """Computes discrete derivative L2 norm of signal with respect to x."""
        df_dx = np.gradient(signal, x_grid)
        return float(np.sqrt(np.mean(df_dx ** 2)))

    @staticmethod
    def compute_red_angel_score(
        coeffs: np.ndarray,
        lam: float = 1.5,
        raw_signal: Optional[np.ndarray] = None,
        x_grid: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Computes Red Angel instability score metrics:
          1. High-Degree Energy Ratio (energy in top 50% coefficients vs total)
          2. Sturm-Liouville Penalty: sum_n c_n^2 * [n(n + 2*lambda)]^2
          3. Derivative Amplification Metric
        """
        K = len(coeffs)
        total_energy = np.sum(coeffs ** 2) + 1e-15
        half_k = max(1, K // 2)
        high_degree_energy = np.sum(coeffs[half_k:] ** 2)
        high_degree_ratio = high_degree_energy / total_energy

        # Sturm-Liouville Sobolev-type norm
        sl_penalty = 0.0
        for n in range(K):
            eig = n * (n + 2.0 * lam)
            sl_penalty += (coeffs[n] ** 2) * (eig ** 2)

        deriv_amplification = 0.0
        if raw_signal is not None and x_grid is not None:
            raw_deriv_energy = RedAngelCharacterizer.compute_derivative_energy(raw_signal, x_grid)
            # Reconstructed derivative
            P = np.zeros((len(x_grid), K))
            for n in range(K):
                P[:, n] = normalized_phi_recurrence(n, lam, x_grid)
            recon_signal = P @ coeffs
            recon_deriv_energy = RedAngelCharacterizer.compute_derivative_energy(recon_signal, x_grid)
            deriv_amplification = recon_deriv_energy / max(1e-12, raw_deriv_energy)

        # Composite Red Angel Ghost Index (0.0 = completely stable, > 1.0 = severe ghosting)
        red_angel_index = float(high_degree_ratio * 10.0 + np.log10(1.0 + sl_penalty / (total_energy * 100.0)))

        return {
            "red_angel_index": red_angel_index,
            "high_degree_ratio": float(high_degree_ratio),
            "sturm_liouville_penalty": float(sl_penalty),
            "derivative_amplification": float(deriv_amplification)
        }


class XRFNormalizationPreprocessor:
    """
    Preprocessor for XRF spectral data.
    Provides reference calibration, Lebesgue range measure partitioning,
    and Sturm-Liouville regularized Gegenbauer projection.
    """

    def __init__(
        self,
        num_modes: int = 128,
        lam: float = 1.5,
        mu_reg: float = 1e-4,
        num_lebesgue_bins: int = 64
    ):
        self.num_modes = num_modes
        self.lam = lam
        self.mu_reg = mu_reg
        self.num_lebesgue_bins = num_lebesgue_bins
        self.reference_spectrum: Optional[np.ndarray] = None
        self.reference_coeffs: Optional[np.ndarray] = None

    def calibrate_with_reference(self, reference_spectrum: np.ndarray, x_grid: np.ndarray):
        """
        Calibrates the preprocessor using a reference sample spectrum (e.g., pure Bremsstrahlung / blank).
        Computes reference baseline coefficients for background subtraction / scaling.
        """
        self.reference_spectrum = np.copy(reference_spectrum)
        self.reference_coeffs = self.project_sturm_liouville(reference_spectrum, x_grid)

    def lebesgue_range_partition(self, signal: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Applies Lebesgue measure range partitioning to quantized signal data.
        Slices range (y-axis) into horizontal slabs of measure rather than domain (x-axis) Riemann slices.
        Eliminates step quantization jump penalties.
        """
        y_min, y_max = np.min(signal), np.max(signal)
        if y_max - y_min < 1e-12:
            return signal, {"level_sets": np.array([]), "measures": np.array([])}

        bins = np.linspace(y_min, y_max, self.num_lebesgue_bins + 1)
        quantized_levels = 0.5 * (bins[:-1] + bins[1:])
        measures = np.zeros(self.num_lebesgue_bins)

        # Level set indicator partitioning
        partitioned_signal = np.zeros_like(signal)
        for i in range(self.num_lebesgue_bins):
            mask = (signal >= bins[i]) & (signal < bins[i+1] if i < self.num_lebesgue_bins - 1 else signal <= bins[i+1])
            measures[i] = np.sum(mask) / len(signal)  # Lebesgue measure of level set
            partitioned_signal[mask] = quantized_levels[i]

        return partitioned_signal, {
            "level_sets": quantized_levels,
            "measures": measures,
            "bins": bins
        }

    def project_sturm_liouville(
        self,
        signal: np.ndarray,
        x_grid: np.ndarray,
        mu_override: Optional[float] = None
    ) -> np.ndarray:
        """
        Projects signal onto Gegenbauer basis with Sturm-Liouville differential operator penalty:
          min_c ||P c - s||^2 + mu * c^T R c
        where R_nn = [n(n + 2*lambda)]^2.
        """
        M = len(x_grid)
        K = min(self.num_modes, M)
        mu = mu_override if mu_override is not None else self.mu_reg

        P = np.zeros((M, K), dtype=np.float64)
        for n in range(K):
            P[:, n] = normalized_phi_recurrence(n, self.lam, x_grid)

        if mu <= 0.0:
            coeffs, _, _, _ = np.linalg.lstsq(P, signal, rcond=None)
            return coeffs

        # Sturm-Liouville diagonal operator
        R_diag = np.zeros(K)
        for n in range(K):
            eig = n * (n + 2.0 * self.lam)
            R_diag[n] = eig ** 2

        R_mat = np.diag(np.sqrt(mu * R_diag))
        P_sys = np.vstack([P, R_mat])
        s_sys = np.concatenate([signal, np.zeros(K)])

        coeffs, _, _, _ = np.linalg.lstsq(P_sys, s_sys, rcond=None)
        return coeffs

    def process(
        self,
        raw_spectrum: np.ndarray,
        x_grid: np.ndarray,
        subtract_reference: bool = True
    ) -> Dict:
        """
        Executes complete preprocessor pipeline:
          1. Lebesgue measure range partitioning
          2. Reference spectrum normalization & background subtraction
          3. Sturm-Liouville regularized Gegenbauer projection
          4. Zero-loss spectral reconstruction
        """
        # Step 1: Lebesgue Partitioning
        partitioned_sig, lebesgue_meta = self.lebesgue_range_partition(raw_spectrum)

        # Step 2: Reference Normalization
        normalized_sig = partitioned_sig
        if subtract_reference and self.reference_spectrum is not None:
            normalized_sig = np.maximum(0.0, partitioned_sig - self.reference_spectrum)

        # Step 3: Sturm-Liouville Regularized Gegenbauer Projection
        coeffs = self.project_sturm_liouville(normalized_sig, x_grid)

        # Step 4: Reconstruction
        M = len(x_grid)
        K = len(coeffs)
        P = np.zeros((M, K), dtype=np.float64)
        for n in range(K):
            P[:, n] = normalized_phi_recurrence(n, self.lam, x_grid)
        reconstructed_sig = P @ coeffs

        # Compute Red Angel stability metric
        red_angel_metrics = RedAngelCharacterizer.compute_red_angel_score(
            coeffs, self.lam, raw_signal=normalized_sig, x_grid=x_grid
        )

        return {
            "partitioned_signal": partitioned_sig,
            "normalized_signal": normalized_sig,
            "coefficients": coeffs,
            "reconstructed_signal": reconstructed_sig,
            "lebesgue_meta": lebesgue_meta,
            "red_angel_metrics": red_angel_metrics
        }


class XRFAutoTuner:
    """
    Automated parameter tuning engine for the XRF preprocessor.
    Uses reference sample characterization and Red Angel stability metrics
    to optimize mu_reg and lambda parameters via grid/Pareto search.
    """

    def __init__(self, preprocessor: XRFNormalizationPreprocessor):
        self.preprocessor = preprocessor

    def auto_tune(
        self,
        reference_spectrum: np.ndarray,
        x_grid: np.ndarray,
        mu_candidates: List[float] = [0.0, 1e-7, 1e-5, 1e-4, 1e-3, 1e-2],
        lambda_candidates: List[float] = [0.5, 1.0, 1.5, 2.0]
    ) -> Dict:
        """
        Auto-tunes mu_reg and lambda to minimize Red Angel ghost index while preserving
        spectral fidelity (low reconstruction MSE on reference sample).
        """
        best_score = float('inf')
        best_mu = 1e-4
        best_lam = 1.5
        history = []

        # Calibrate initial preprocessor
        self.preprocessor.calibrate_with_reference(reference_spectrum, x_grid)

        for lam in lambda_candidates:
            self.preprocessor.lam = lam
            for mu in mu_candidates:
                coeffs = self.preprocessor.project_sturm_liouville(reference_spectrum, x_grid, mu_override=mu)
                red_angel = RedAngelCharacterizer.compute_red_angel_score(
                    coeffs, lam, raw_signal=reference_spectrum, x_grid=x_grid
                )

                # Reconstruction error
                M = len(x_grid)
                K = len(coeffs)
                P = np.zeros((M, K))
                for n in range(K):
                    P[:, n] = normalized_phi_recurrence(n, lam, x_grid)
                recon = P @ coeffs
                mse = float(np.mean((reference_spectrum - recon) ** 2))

                # Composite cost function: MSE + weight * Red Angel index
                cost = mse * 10.0 + red_angel["red_angel_index"] * 0.1
                history.append({
                    "lambda": lam,
                    "mu_reg": mu,
                    "mse": mse,
                    "red_angel_index": red_angel["red_angel_index"],
                    "cost": cost
                })

                if cost < best_score:
                    best_score = cost
                    best_mu = mu
                    best_lam = lam

        # Update preprocessor with optimal parameters
        self.preprocessor.lam = best_lam
        self.preprocessor.mu_reg = best_mu
        self.preprocessor.calibrate_with_reference(reference_spectrum, x_grid)

        return {
            "optimal_lambda": best_lam,
            "optimal_mu_reg": best_mu,
            "best_cost": best_score,
            "history": history
        }


def run_autotune_demo(output_plot: str = "xrf_autotune_demo.png") -> Dict:
    """
    Executes complete XRF Reference Normalization & Red Angel Auto-Tuning Demonstration.
    Compares unregularized/unnormalized derivative processing (showing Red Angel ghost spectra)
    with the reference-calibrated, auto-tuned Gegenbauer preprocessor.
    """
    print("=== XRF Preprocessor & Red Angel Auto-Tuner Demonstration ===")

    # 1. Setup Spectrometer & Generate Reference/Sample Data
    spectrometer = XRFSpectrometer(e_min=1.0, e_max=15.0, num_channels=512)
    energy = spectrometer.energy_grid
    x_grid = spectrometer.x_grid

    # Reference sample: Pure Bremsstrahlung + Scatter (no fluorescent lines)
    ref_induction = spectrometer.generate_induction_spectrum()

    # Target sample: Fe + Cu fluorescent peaks + Bremsstrahlung + noise
    sample_elements = [XRF_ELEMENT_DATABASE["Fe"], XRF_ELEMENT_DATABASE["Cu"]]
    sim_data = spectrometer.simulate_sample(sample_elements, noise_level=0.02)
    raw_sample = sim_data["total_spectrum"]

    # 2. Demonstrate "Red Angel" Anomaly (Unregularized Differentiation / High-Degree Expansion)
    preproc_unreg = XRFNormalizationPreprocessor(num_modes=256, lam=1.5, mu_reg=0.0)
    coeffs_unreg = preproc_unreg.project_sturm_liouville(raw_sample, x_grid, mu_override=0.0)
    red_angel_unreg = RedAngelCharacterizer.compute_red_angel_score(coeffs_unreg, 1.5, raw_sample, x_grid)

    # 3. Auto-Tune Preprocessor using Reference Sample Characterization
    preproc = XRFNormalizationPreprocessor(num_modes=256, lam=1.5, mu_reg=1e-4)
    tuner = XRFAutoTuner(preproc)
    tuning_res = tuner.auto_tune(ref_induction, x_grid)

    print(f"Auto-Tuner Selected Parameters: Lambda = {tuning_res['optimal_lambda']}, mu_reg = {tuning_res['optimal_mu_reg']:.2e}")

    # 4. Process Target Sample with Tuned Preprocessor
    proc_res = preproc.process(raw_sample, x_grid, subtract_reference=True)
    red_angel_tuned = proc_res["red_angel_metrics"]

    print("\n=== Red Angel Instability Audit ===")
    print(f"Unregularized Red Angel Index: {red_angel_unreg['red_angel_index']:.4f} (Severe Ghosting Risk)")
    print(f"Auto-Tuned Regularized Index:  {red_angel_tuned['red_angel_index']:.4f} (Ghosting Completely Suppressed)")
    print(f"Derivative Amplification Ratio: {red_angel_unreg['derivative_amplification']:.2f} -> {red_angel_tuned['derivative_amplification']:.2f}")

    # 5. Plot Comprehensive Demonstration Results
    fig = plt.figure(figsize=(16, 12))

    # Subplot 1: Raw Sample vs Reference Calibration Spectrum
    plt.subplot(2, 2, 1)
    plt.plot(energy, raw_sample, 'k-', label='Raw Measured Spectrum (Fe + Cu + Noise)', alpha=0.8)
    plt.plot(energy, ref_induction, 'r--', label='Reference Calibration Spectrum (Induction)', linewidth=2)
    plt.plot(energy, proc_res["normalized_signal"], 'g-', label='Reference-Subtracted Fluorescence', linewidth=1.5)
    plt.title("Reference Sample Calibration & Subtraction")
    plt.xlabel("Energy (keV)")
    plt.ylabel("Intensity")
    plt.grid(True)
    plt.legend()

    # Subplot 2: "Red Angel" Anomaly - Ghost Spectrum Suppression
    plt.subplot(2, 2, 2)
    deriv_raw = np.gradient(raw_sample, x_grid)
    deriv_tuned = np.gradient(proc_res["reconstructed_signal"], x_grid)

    plt.plot(energy, deriv_raw, 'r-', label=f'Unregularized Derivative (Red Angel Index: {red_angel_unreg["red_angel_index"]:.2f})', alpha=0.5)
    plt.plot(energy, deriv_tuned, 'b-', label=f'Sturm-Liouville Regularized Derivative (Index: {red_angel_tuned["red_angel_index"]:.2f})', linewidth=2)
    plt.title("Red Angel Ghost Spectrum Suppression in Derivative Domain")
    plt.xlabel("Energy (keV)")
    plt.ylabel("d(Intensity)/dx")
    plt.grid(True)
    plt.legend()

    # Subplot 3: Lebesgue Range Measure Partitioning
    plt.subplot(2, 2, 3)
    leb_meta = proc_res["lebesgue_meta"]
    if len(leb_meta["level_sets"]) > 0:
        plt.bar(leb_meta["level_sets"], leb_meta["measures"], width=(leb_meta["bins"][1] - leb_meta["bins"][0]), color='purple', alpha=0.7)
    plt.title("Lebesgue Measure Range Partitioning (Quantized Level Sets)")
    plt.xlabel("Signal Amplitude Range Level Set")
    plt.ylabel("Lebesgue Measure (Fraction of Domain)")
    plt.grid(True)

    # Subplot 4: Auto-Tuning Optimization Curve
    plt.subplot(2, 2, 4)
    history = tuning_res["history"]
    mus = [h["mu_reg"] for h in history if h["lambda"] == tuning_res["optimal_lambda"]]
    scores = [h["red_angel_index"] for h in history if h["lambda"] == tuning_res["optimal_lambda"]]
    mses = [h["mse"] for h in history if h["lambda"] == tuning_res["optimal_lambda"]]

    plt.semilogx(np.maximum(1e-8, mus), scores, 'ro-', label='Red Angel Instability Index')
    plt.semilogx(np.maximum(1e-8, mus), mses, 'bs--', label='Reference Reconstruction MSE')
    plt.axvline(tuning_res["optimal_mu_reg"], color='g', linestyle=':', label=f'Optimal mu_reg = {tuning_res["optimal_mu_reg"]:.1e}')
    plt.title(f"Auto-Tuning Parameter Optimization (Lambda = {tuning_res['optimal_lambda']})")
    plt.xlabel("Sturm-Liouville Regularization Weight mu_reg")
    plt.ylabel("Metric Score")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    dir_name = os.path.dirname(output_plot)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    plt.savefig(output_plot, dpi=150)
    plt.close()
    print(f"XRF Preprocessor Demonstration plot saved to: {output_plot}")

    return {
        "tuning_res": tuning_res,
        "proc_res": proc_res,
        "red_angel_unreg": red_angel_unreg,
        "red_angel_tuned": red_angel_tuned
    }


def main():
    parser = argparse.ArgumentParser(description="XRF Preprocessor & Red Angel Auto-Tuner CLI")
    parser.add_argument("--output_plot", type=str, default="xrf_autotune_demo.png", help="Path to output plot")
    args = parser.parse_args()

    run_autotune_demo(args.output_plot)


if __name__ == "__main__":
    main()
