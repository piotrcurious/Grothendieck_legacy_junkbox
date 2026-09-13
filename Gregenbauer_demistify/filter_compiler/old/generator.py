import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import warnings

# ==============================================================================
# Helper functions
# ==============================================================================

def gegenbauer_c(n, lmb, x):
    """
    Evaluate Gegenbauer polynomial C_n^lambda(x) at x using recurrence.
    Dtype-safe and vectorized for numpy arrays.
    """
    x = np.asarray(x, dtype=float)
    if n == 0:
        return np.ones_like(x, dtype=float)
    if n == 1:
        return 2.0 * lmb * x

    c_nm2 = np.ones_like(x, dtype=float)
    c_nm1 = 2.0 * lmb * x

    for k in range(2, n + 1):
        c_n = (2.0 * (k + lmb - 1.0) * x * c_nm1 - (k + 2.0 * lmb - 2.0) * c_nm2) / k
        c_nm2 = c_nm1
        c_nm1 = c_n
    return c_nm1

def eval_gegenbauer_window(coeff, lmb, x):
    """
    Evaluate fitted Gegenbauer window at x in [-1, 1].
    Smooth and exact polynomial evaluation without hard clipping.
    Dtype-safe.
    """
    x = np.asarray(x, dtype=float)
    w = np.zeros_like(x, dtype=float)
    for k, c in enumerate(coeff):
        degree = 2 * k
        w += c * gegenbauer_c(degree, lmb, x)
    return w

def fit_gegenbauer_window(basis_count, lmb, fit_samples=100, target_type='hann'):
    """
    Fit even Gegenbauer basis terms C_0, C_2, ..., C_{2*(B-1)} to a target window.
    Uses Gauss-Gegenbauer-like Chebyshev nodes and Weighted Least Squares (WLS).
    """
    # 1. Use Chebyshev nodes of the first kind to prevent Runge-like oscillations
    u = np.cos((2 * np.arange(fit_samples) + 1) / (2 * fit_samples) * np.pi)

    # 2. Target window shape
    if target_type == 'hann':
        target = 0.5 * (1.0 + np.cos(np.pi * u))
    elif target_type == 'hamming':
        target = 0.54 + 0.46 * np.cos(np.pi * u)
    elif target_type == 'blackman':
        target = 0.42 + 0.5 * np.cos(np.pi * u) + 0.08 * np.cos(2.0 * np.pi * u)
    else:
        target = 0.5 * (1.0 + np.cos(np.pi * u))  # Fallback

    # 3. Construct Design Matrix A
    A = np.zeros((fit_samples, basis_count), dtype=float)
    for k in range(basis_count):
        degree = 2 * k
        A[:, k] = gegenbauer_c(degree, lmb, u)

    # 4. Apply Weighted Least Squares using Gegenbauer orthogonality weight: (1 - u^2)^{lambda - 0.5}
    weight = (1.0 - u**2) ** (lmb - 0.5)
    # Clip any potential extremely high values at boundary nodes just in case
    weight = np.clip(weight, 1e-5, 1e5)
    sqrt_W = np.sqrt(weight)

    A_weighted = A * sqrt_W[:, np.newaxis]
    target_weighted = target * sqrt_W

    # Solve least-squares system
    coeff, _, _, _ = np.linalg.lstsq(A_weighted, target_weighted, rcond=None)

    # 5. Normalize so window is exactly 1.0 at center (u = 0)
    w0 = 0.0
    for k in range(basis_count):
        degree = 2 * k
        w0 += coeff[k] * gegenbauer_c(degree, lmb, 0.0)

    if np.abs(w0) < 1e-4:
        warnings.warn(f"Fitting may be unstable: w(0) is close to zero ({w0:.6f}). Try adjusting lambda_param or basis_count.")

    if np.abs(w0) < 1e-12:
        w0 = 1.0
    coeff /= w0

    # Fitting Quality Check
    fitted_win = eval_gegenbauer_window(coeff, lmb, u)
    mse = np.mean((target - fitted_win)**2)
    if mse > 1e-3:
        warnings.warn(f"Poor window fit detected (MSE: {mse:.6f}). The selected basis_count ({basis_count}) may be underfitting. Consider increasing N or basis_count.")

    # Return u sorted for clean plotting
    sort_idx = np.argsort(u)
    return coeff, u[sort_idx], target[sort_idx]

def design_approx_qmf_filters(N, cutoff, lmb, coeff):
    """
    Design Lowpass (h0) and Highpass (h1) approximate QMF filters using the fitted window.
    Note: These are windowed-sinc approximate complementary pairs.
    """
    mid = (N - 1) / 2.0
    h0 = np.zeros(N, dtype=float)

    for n in range(N):
        m = n - mid
        x = 2.0 * n / (N - 1) - 1.0
        w = eval_gegenbauer_window(coeff, lmb, x)

        if np.abs(m) < 1e-12:
            ideal = 2.0 * cutoff
        else:
            ideal = np.sin(2.0 * np.pi * cutoff * m) / (np.pi * m)
        h0[n] = ideal * w

    # Normalize h0 so DC gain = 1
    sum_h0 = np.sum(h0)
    if np.abs(sum_h0) < 1e-12:
        sum_h0 = 1.0
    h0 /= sum_h0

    # Complementary mirror highpass: h1[n] = (-1)^n * h0[n]
    h1 = np.zeros(N, dtype=float)
    for n in range(N):
        sign = -1.0 if (n % 2 != 0) else 1.0
        h1[n] = sign * h0[n]

    return h0, h1

# ==============================================================================
# Main Generation
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Gegenbauer Polynomial QMF Filter Design & Code Generator")
    parser.add_argument("--N", type=int, default=63, help="Filter length (odd recommended)")
    parser.add_argument("--cutoff", type=float, default=0.25, help="Normalized low-pass cutoff (0.0 to 0.5, fs/4 = 0.25)")
    parser.add_argument("--lambda_param", type=float, default=1.25, help="Gegenbauer shape parameter (lambda > -0.5)")
    parser.add_argument("--basis_count", type=int, default=None, help="Number of even Gegenbauer basis terms (default scaled based on N)")
    parser.add_argument("--target", type=str, default="hann", choices=["hann", "hamming", "blackman"], help="Target window shape")
    parser.add_argument("--sampling_rate", type=int, default=2000, help="Sampling rate in Hz (default: 2000)")
    parser.add_argument("--output_header", type=str, default="gegenbauer_coeffs.h", help="Path to generate the Arduino include header file")
    parser.add_argument("--output_plot", type=str, default="gegenbauer_response.png", help="Path to generate response visualization plot")

    args = parser.parse_args()

    # Parameter Validation
    if args.lambda_param <= -0.5:
        raise ValueError(f"Gegenbauer parameter lambda_param must be strictly > -0.5. Received: {args.lambda_param}")

    # Heuristic dynamic basis scaling based on N if not specified
    if args.basis_count is None:
        args.basis_count = max(4, int(np.sqrt(args.N)) + 1)
        print(f"Using dynamically scaled basis_count: {args.basis_count} for filter length N: {args.N}")

    # 1. Fit Window
    win_coeffs, u, target_win = fit_gegenbauer_window(args.basis_count, args.lambda_param, target_type=args.target)

    # 2. Design Filters
    h0, h1 = design_approx_qmf_filters(args.N, args.cutoff, args.lambda_param, win_coeffs)

    # 3. Scale for Fixed-Point Q15-style conversion (32767.0 base) with saturating cast
    scale = 32767.0
    h0_fixed = np.clip(np.round(h0 * scale), -32768, 32767).astype(np.int16)
    h1_fixed = np.clip(np.round(h1 * scale), -32768, 32767).astype(np.int16)

    # 4. Generate C++ / Arduino Header with modern constexpr
    header_content = f"""// Auto-generated Gegenbauer QMF Coefficients Header
// Designed using Gegenbauer Polynomial-Transform Fit
// Parameters:
//   Filter Length (N): {args.N}
//   Cutoff Frequency: {args.cutoff} (normalized to Nyquist=0.5)
//   Gegenbauer Lambda: {args.lambda_param}
//   Basis Count: {args.basis_count}
//   Target Window Type: {args.target}
//   Sampling Rate (Hz): {args.sampling_rate}

#ifndef GEGENBAUER_COEFFS_H
#define GEGENBAUER_COEFFS_H

#include <Arduino.h>

// Modern self-describing metadata constexpr declarations
constexpr int GEG_N = {args.N};
constexpr int GEG_SAMPLING_RATE = {args.sampling_rate};
constexpr float GEG_CUTOFF = {args.cutoff}f;
constexpr float GEG_LAMBDA = {args.lambda_param}f;
constexpr int GEG_BASIS_COUNT = {args.basis_count};

// Float filter coefficients
const float h_geg_float[GEG_N] = {{
    {", ".join(f"{val:.8f}f" for val in h0)}
}};

const float g_geg_float[GEG_N] = {{
    {", ".join(f"{val:.8f}f" for val in h1)}
}};

// 16-bit fixed point coefficients (scaled by 2^15 - 1 = 32767)
const int16_t h_geg_fixed[GEG_N] PROGMEM = {{
    {", ".join(str(val) for val in h0_fixed)}
}};

const int16_t g_geg_fixed[GEG_N] PROGMEM = {{
    {", ".join(str(val) for val in h1_fixed)}
}};

#endif // GEGENBAUER_COEFFS_H
"""

    header_dir = os.path.dirname(args.output_header)
    if header_dir and not os.path.exists(header_dir):
        os.makedirs(header_dir, exist_ok=True)

    with open(args.output_header, "w") as f:
        f.write(header_content)
    print(f"Successfully generated Arduino coefficients header at: {args.output_header}")

    # 5. Mathematically Rigorous Complex FFT on a [0, 2pi) Grid
    K = 4096
    H0 = np.fft.fft(h0, K)
    H1 = np.fft.fft(h1, K)

    # Shift by pi corresponds precisely to shifting by K // 2 bins
    H0_shift = np.roll(H0, K // 2)
    H1_shift = np.roll(H1, K // 2)

    # QMF Metrics (Power Complementarity & True Alias Distortion)
    power_complementarity = np.abs(H0)**2 + np.abs(H1)**2
    # Alias Transfer Function: A(z) = 0.5 * [H0(z)H0(-z) + H1(z)H1(-z)]
    aliasing_func = 0.5 * np.abs(H0 * H0_shift + H1 * H1_shift)

    # Slice to nonnegative frequencies (first half, [0, pi])
    half_K = K // 2
    f_normalized = np.linspace(0, 0.5, half_K)
    pow_db = 10 * np.log10(power_complementarity[:half_K] + 1e-15)
    aliasing_db = 20 * np.log10(aliasing_func[:half_K] + 1e-15)

    # 6. Generate Plot with comprehensive DSP verification subplots
    plt.figure(figsize=(16, 12))

    # Plot 1: Window Approximation
    plt.subplot(2, 2, 1)
    fitted_win = eval_gegenbauer_window(win_coeffs, args.lambda_param, u)
    plt.plot(u, target_win, 'r--', label=f'Target ({args.target.capitalize()})', linewidth=2)
    plt.plot(u, fitted_win, 'b-', label='Gegenbauer WLS Window', linewidth=2)
    plt.title(f"WLS Window Fitting using {args.basis_count} Gegenbauer Polynomials (Lambda={args.lambda_param})")
    plt.xlabel("u")
    plt.ylabel("Weight")
    plt.grid(True)
    plt.legend()

    # Plot 2: Lowpass & Highpass Responses (dB)
    plt.subplot(2, 2, 2)
    plt.plot(f_normalized, 20 * np.log10(np.abs(H0[:half_K]) + 1e-12), 'b-', label='H0 (Lowpass)', linewidth=2)
    plt.plot(f_normalized, 20 * np.log10(np.abs(H1[:half_K]) + 1e-12), 'r-', label='H1 (Highpass)', linewidth=2)
    plt.axvline(args.cutoff, color='g', linestyle=':', label=f'Cutoff = {args.cutoff}')
    plt.title(f"Frequency Response of Designed Filters (N={args.N})")
    plt.xlabel("Normalized Frequency (Cycles/Sample)")
    plt.ylabel("Magnitude (dB)")
    plt.ylim(-80, 5)
    plt.grid(True)
    plt.legend()

    # Plot 3: Power Complementarity & Aliasing Distortion (Approximate QMF Metrics)
    plt.subplot(2, 2, 3)
    plt.plot(f_normalized, pow_db, 'g-', label='Power Complementarity $|H_0|^2 + |H_1|^2$', linewidth=2)
    plt.plot(f_normalized, aliasing_db, 'm--', label='Alias Transfer Function $A(e^{j\\omega})$', linewidth=1.5)
    plt.axhline(0, color='r', linestyle='--', alpha=0.5)
    plt.title("Approximate QMF Metrics")
    plt.xlabel("Normalized Frequency")
    plt.ylabel("Magnitude (dB)")
    plt.ylim(-60, 10)
    plt.grid(True)
    plt.legend()

    # Plot 4: Filter Coefficients Impulse Responses
    plt.subplot(2, 2, 4)
    plt.stem(range(args.N), h0, linefmt='b-', markerfmt='bo', basefmt='r-', label='h0[n]')
    plt.stem(range(args.N), h1, linefmt='r--', markerfmt='rx', basefmt='r-', label='h1[n]')
    plt.title("Filter Coefficients (Impulse Response)")
    plt.xlabel("n")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(args.output_plot, dpi=150)
    plt.close()
    print(f"Successfully generated transfer function and window plots at: {args.output_plot}")

if __name__ == "__main__":
    main()
