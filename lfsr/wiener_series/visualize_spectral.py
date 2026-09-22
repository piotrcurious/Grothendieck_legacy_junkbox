#!/usr/bin/env python3
import subprocess
import json
import os
import numpy as np
import matplotlib.pyplot as plt

def run_cpp_cli(L, poly, beta, output_json):
    cli_path = os.path.join(os.path.dirname(__file__), "lfsr_wiener_cli")
    cmd = [cli_path, "-L", str(L), "-p", str(poly), "-b", str(beta), "-o", output_json]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    with open(output_json, 'r') as f:
        data = json.load(f)
    return data

def plot_spectral_properties(data, save_prefix="lfsr_spectral"):
    L = data["L"]
    N = data["N"]
    bipolar = np.array(data["bipolar"])
    power_spectrum = np.array(data["power_spectrum"])
    autocorr = np.array(data["autocorrelation"])
    wiener_energy = np.array(data["wiener_energy_distribution"])

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f"LFSR Spectral & Discrete Wiener Chaos Analysis (L={L}, N={N})", fontsize=14, fontweight='bold')

    # 1. Bipolar Time-Domain Sequence u_n = (-1)^x_n
    ax1 = axes[0, 0]
    ax1.step(np.arange(N), bipolar, where='mid', color='navy', linewidth=2, label=r'$u_n = (-1)^{x_n}$')
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.6)
    ax1.set_title("Time-Domain Bipolar Sequence", fontsize=11)
    ax1.set_xlabel("Time Index $n$")
    ax1.set_ylabel("Amplitude $u_n$")
    ax1.set_ylim(-1.5, 1.5)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. Power Spectrum |U_k|^2
    ax2 = axes[0, 1]
    k_indices = np.arange(N)
    ax2.stem(k_indices, power_spectrum, linefmt='b-', markerfmt='bo', basefmt='r-')
    ax2.axhline(2**L, color='red', linestyle='--', linewidth=1.5, label=f'Flat Level $2^L = {2**L}$')
    ax2.set_title("Discrete Fourier Power Spectrum $|U_k|^2$", fontsize=11)
    ax2.set_xlabel("Frequency Index $k$")
    ax2.set_ylabel("Power Spectrum")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 3. Two-Valued Autocorrelation R(d)
    ax3 = axes[0, 2]
    ax3.stem(k_indices, autocorr, linefmt='g-', markerfmt='go', basefmt='r-')
    ax3.axhline(-1.0, color='darkred', linestyle='--', label=r'Off-Peak Level $-1$')
    ax3.axhline(N, color='green', linestyle=':', label=f'Peak Level $N = {N}$')
    ax3.set_title(r"Periodic Autocorrelation $R(d) = \sum_n u_n u_{n+d}$", fontsize=11)
    ax3.set_xlabel("Delay $d$")
    ax3.set_ylabel("$R(d)$")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # 4. Gauss Sum & DFT Magnitude Comparison
    ax4 = axes[1, 0]
    dft_mags = np.sqrt(power_spectrum)
    ax4.plot(k_indices[1:], dft_mags[1:], 'o-', color='purple', label=r'$|U_k| = |g(\chi_k, \psi)|$')
    ax4.axhline(2**(L/2.0), color='orange', linestyle='--', linewidth=1.5, label=f'Theoretical $2^{{L/2}} = {2**(L/2.0):.3f}$')
    ax4.set_title("Gauss Sum / Fourier Magnitudes $|U_k|$", fontsize=11)
    ax4.set_xlabel("Frequency Index $k > 0$")
    ax4.set_ylabel("Magnitude $|U_k|$")
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    # 5. Discrete Wiener Chaos Energy Distribution E(k)
    ax5 = axes[1, 1]
    degrees = np.arange(len(wiener_energy))
    ax5.bar(degrees, wiener_energy, color='teal', alpha=0.8, edgecolor='black', width=0.5)
    ax5.set_title("Wiener Chaos Energy Spectrum $E(k)$", fontsize=11)
    ax5.set_xlabel("Chaos Degree $k$")
    ax5.set_ylabel("Chaos Energy $E(k)$")
    ax5.set_xticks(degrees)
    ax5.grid(True, alpha=0.3)

    # 6. Non-linear Filtering Wiener Chaos Comparison
    ax6 = axes[1, 2]
    # Comparison of Chaos Energy between Linear Trace vs Nonlinear Filter (Product u0*u1*u2)
    nonlinear_chaos_energy = np.zeros(L + 1)
    nonlinear_chaos_energy[L] = 1.0 # Pure degree-L chaos
    width = 0.35
    ax6.bar(degrees - width/2, wiener_energy, width, label='Linear Trace $x_n$', color='teal', alpha=0.8)
    ax6.bar(degrees + width/2, nonlinear_chaos_energy, width, label='Nonlinear Filter $u_0 u_1 u_2$', color='crimson', alpha=0.8)
    ax6.set_title("Wiener Chaos Subspace Allocation", fontsize=11)
    ax6.set_xlabel("Chaos Degree $k$")
    ax6.set_ylabel("Energy $E(k)$")
    ax6.set_xticks(degrees)
    ax6.grid(True, alpha=0.3)
    ax6.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = f"{save_prefix}_L{L}.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Plot successfully saved to {plot_path}")
    return plot_path

def main():
    json_path = os.path.join(os.path.dirname(__file__), "spectral_report_L3.json")
    data = run_cpp_cli(L=3, poly=11, beta=1, output_json=json_path)
    plot_spectral_properties(data, save_prefix=os.path.join(os.path.dirname(__file__), "lfsr_spectral"))

if __name__ == "__main__":
    main()
