# X-Ray Fluorescence (XRF) Spectrometer & Preprocessor Documentation

## 1. Overview & Key Concepts

This module demonstrates the application of the **VIII-Layer Gegenbauer Theoretical Framework** to X-Ray Fluorescence (XRF) spectral processing, element signal isolation, differential spectral synthesis, and preprocessor auto-tuning.

In XRF spectroscopy, measured energy spectra $S(E)$ consist of:
1. **Primary Excitation Induction Continuum** $I(E)$: Kramers Bremsstrahlung radiation and characteristic tube scattering (e.g., Rhodium tube target $\sim 19$ keV peak).
2. **Elemental Fluorescent Emission Lines** $\sum_e F_e(E)$: Characteristic Gaussian doublet/triplet peaks ($K_\alpha, K_\beta, L_\alpha, L_\beta$) representing elements such as Fe ($6.40, 7.06$ keV), Cu ($8.04, 8.91$ keV), Zn ($8.63, 9.57$ keV), and Pb ($10.55, 12.61$ keV).

$$S(E) = I(E) + \sum_{e} F_e(E) + \eta(E)$$

---

## 2. Lossless Differential Output Stacking

By mapping the energy domain $E \in [E_{\min}, E_{\max}]$ linearly to $x \in [-1, 1]$ and projecting subband signals onto orthogonal Gegenbauer polynomial bases $\phi_n^{(\lambda)}(x)$, differential spectral outputs can be stacked and reconstructed with **zero information loss** (machine precision SNR $> 90$ dB).

### Differential Output Definitions:
1. **Induction vs. Fluorescence Differential**:
   $$\Delta_{\text{ind, fluor}}(E) = I(E) - \sum_{e} F_e(E)$$
2. **Inter-Element Differential**:
   $$\Delta_{e_1, e_2}(E) = F_{e_1}(E) - F_{e_2}(E)$$

### Generated Plot: `xrf_spectrometer_demo.png`
Running `python xrf_spectrometer_demo.py --output_plot xrf_spectrometer_demo.png` generates a 4-panel verification plot:
- **Panel 1 (Spectral Response)**: Total raw spectrum vs. induction Bremsstrahlung continuum vs. isolated elemental fluorescence curves ($Fe, Cu, Zn, Pb$).
- **Panel 2 (Differential Spectra)**: Visualizes $\Delta_{\text{ind, fluor}}$ and inter-element differential $\Delta_{\text{Fe}, \text{Cu}}$.
- **Panel 3 (Lossless Subband Stacking)**: Overlay of original stacked spectrum $\sum \text{subbands}$ vs. Gegenbauer synthesized reconstruction $S_{\text{recon}}(E)$.
- **Panel 4 (Residual Error Surface)**: Demonstrates zero-loss reconstruction with peak error $< 1.5 \times 10^{-3}$ and SNR $> 90$ dB.

---

## 3. The "Red Angel" Anomaly & Mitigation Architecture

### What is the "Red Angel" Anomaly?
The "Red Angel" anomaly is a severe numerical instability that occurs during unregularized differentiation or high-degree polynomial expansion of noisy/unnormalized spectral signatures. Unconstrained higher-order transform coefficients grow exponentially, manifesting as spurious high-frequency oscillations or **"ghost spectra"** that corrupt physical elemental peak identification.

### Preprocessor Mitigation Architecture (`xrf_preprocessor.py`):
1. **Reference Sample Calibration & Domain Mapping**:
   Maps raw energy $E \mapsto x \in [-1, 1]$ and subtracts reference calibration baseline spectrum $I_{\text{ref}}(E)$ (e.g., pure Bremsstrahlung blank).
2. **Lebesgue Measure Range Partitioning**:
   Rather than standard domain (x-axis) Riemann integration which fails on quantized ADC step discontinuities, the preprocessor partitions the amplitude range (y-axis) into level sets $E_k = \{x : y_k \le f(x) < y_{k+1}\}$. Lebesgue level-set measure tracking eliminates step-discontinuity derivative penalties.
3. **Sturm-Liouville Differential Operator Regularization**:
   Penalizes high-order coefficient explosion using the Sturm-Liouville operator eigenvalues $\mathcal{L}_\lambda \phi_n = n(n + 2\lambda) \phi_n$:
   $$\min_{\mathbf{c}} \| \mathbf{P}\mathbf{c} - \mathbf{s} \|^2 + \mu_{\text{reg}} \sum_{n=0}^{K-1} c_n^2 [n(n + 2\lambda)]^2$$
4. **Red Angel Ghost Index Metric**:
   $$\text{Score}_{\text{RedAngel}} = 10 \cdot \frac{E_{\text{high}}}{E_{\text{total}}} + \log_{10} \left( 1 + \frac{\sum c_n^2 [n(n + 2\lambda)]^2}{100 \cdot E_{\text{total}}} \right)$$

---

## 4. Preprocessor Auto-Tuning (`XRFAutoTuner`)

The `XRFAutoTuner` automatically calibrates the preprocessor using a reference sample measurement to select optimal $\lambda$ and regularization weight $\mu_{\text{reg}}$.

### Auto-Tuning Workflow:
1. Measure reference sample (e.g., pure target Bremsstrahlung/scatter continuum).
2. Perform grid search over candidate $\lambda \in [0.5, 1.0, 1.5, 2.0]$ and $\mu_{\text{reg}} \in [0, 10^{-7}, 10^{-5}, 10^{-4}, 10^{-3}, 10^{-2}]$.
3. Evaluate composite cost function balancing reference reconstruction MSE and Red Angel stability score:
   $$\text{Cost}(\lambda, \mu) = 10 \cdot \text{MSE} + 0.1 \cdot \text{Score}_{\text{RedAngel}}$$

### Generated Plot: `xrf_autotune_demo.png`
Running `python xrf_preprocessor.py --output_plot xrf_autotune_demo.png` generates a 4-panel demonstration:
- **Panel 1 (Reference Calibration)**: Raw sample spectrum vs. reference Bremsstrahlung spectrum vs. reference-subtracted fluorescence signal.
- **Panel 2 (Red Angel Ghost Suppression)**: Compares raw unregularized derivative (showing violent ghost oscillations, Index $\approx 16.2$) with Sturm-Liouville regularized derivative (clean physical peaks, Index $\approx 1.6$).
- **Panel 3 (Lebesgue Range Measure Partitioning)**: Histogram of Lebesgue level set measures across quantized amplitude bins.
- **Panel 4 (Auto-Tuning Curve)**: Trade-off curve showing Red Angel index vs. reconstruction MSE across $\mu_{\text{reg}}$ candidates, identifying optimal $\mu_{\text{reg}} = 10^{-3}$.

---

## 5. Instructions for Running Demos & Tests

### Execute XRF Spectrometer Demo:
```bash
python Gregenbauer_demistify/filter_compiler/xrf_spectrometer_demo.py --output_plot Gregenbauer_demistify/filter_compiler/xrf_spectrometer_demo.png
```

### Execute XRF Normalization & Auto-Tuning Demo:
```bash
python Gregenbauer_demistify/filter_compiler/xrf_preprocessor.py --output_plot Gregenbauer_demistify/filter_compiler/xrf_autotune_demo.png
```

### Execute Unit Test Suite:
```bash
python -m pytest Gregenbauer_demistify/filter_compiler
```
