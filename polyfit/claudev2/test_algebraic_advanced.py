import subprocess
import os
import numpy as np
import matplotlib.pyplot as plt

def compile_cpp():
    cmd = [
        "g++",
        "-I", ".",
        "polyfit/claudev2/src/arduino_polyfit.cpp",
        "polyfit/claudev2/src/mock_arduino.cpp",
        "polyfit/claudev2/src/main.cpp",
        "polyfit/claudev2/test_case.cpp",
        "-o", "polyfit/claudev2/test_bin"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Compilation failed!")
        print(result.stderr)
        return False
    return True

def run_advanced_algebraic_test(x_val, degree):
    cpp_code = f"""
#include "polyfit/claudev2/src/arduino_polyfit.hpp"
#include "polyfit/claudev2/src/mock_arduino.hpp"

using namespace polyfit;

void setup() {{
    CategoricalFeatureExtractor extractor({degree});
    float mono[{degree} + 1];
    float frob[{degree}];
    float cyc[{degree} * 2];

    extractor.extract({x_val}f, mono);
    extractor.extract_frobenius({x_val}f, frob);
    extractor.extract_cyclotomic({x_val}f, cyc);

    Serial.println("--- Advanced Algebraic Features ---");
    Serial.print("Monomials: ");
    for(int i=0; i<={degree}; ++i) {{ Serial.print(mono[i]); Serial.print(" "); }}
    Serial.println("");

    Serial.print("Frobenius: ");
    for(int i=0; i<{degree}; ++i) {{ Serial.print(frob[i]); Serial.print(" "); }}
    Serial.println("");

    Serial.print("Cyclotomic: ");
    for(int i=0; i<{degree}*2; ++i) {{ Serial.print(cyc[i]); Serial.print(" "); }}
    Serial.println("");
}}

void loop() {{}}
"""
    with open("polyfit/claudev2/test_case.cpp", 'w') as f:
        f.write(cpp_code)

    if not compile_cpp(): return None

    result = subprocess.run(["./polyfit/claudev2/test_bin"], capture_output=True, text=True)

    output = {}
    for line in result.stdout.splitlines():
        if "Monomials:" in line:
            output['monomials'] = [float(f) for f in line.split("Monomials:")[1].strip().split()]
        if "Frobenius:" in line:
            output['frobenius'] = [float(f) for f in line.split("Frobenius:")[1].strip().split()]
        if "Cyclotomic:" in line:
            output['cyclotomic'] = [float(f) for f in line.split("Cyclotomic:")[1].strip().split()]
    return output

def main():
    x_range = np.linspace(0.1, 10.0, 100)
    degree = 4

    results = []
    for x in x_range:
        results.append(run_advanced_algebraic_test(x, degree))

    # Extract features for plotting
    monos = np.array([r['monomials'] for r in results])
    frobs = np.array([r['frobenius'] for r in results])
    cycs = np.array([r['cyclotomic'] for r in results])

    # Advanced Test: Signal Representation using concatenated Algebraic Basis
    # Signal: a noisy combination of algebraic features
    y_true = 0.5 * monos[:, 1] + 0.3 * frobs[:, 1] + 0.2 * cycs[:, 0]
    y_noisy = y_true + np.random.normal(0, 0.05, size=y_true.shape)

    # Concatenate all features into a large basis matrix
    # [1, mono1..4, frob0..3, cyc0..7]
    X_basis = np.hstack([monos, frobs, cycs])

    # Fit using pseudo-inverse (least squares)
    weights = np.linalg.pinv(X_basis) @ y_noisy
    y_pred = X_basis @ weights

    fig, axs = plt.subplots(4, 1, figsize=(12, 24))

    # Plot Monomials
    for i in range(1, degree + 1):
        axs[0].plot(x_range, monos[:, i], label=f'Monomial deg {i}')
    axs[0].set_title("Categorical Monomial Features (F2 Polynomial Mul)")
    axs[0].legend()
    axs[0].grid(True)

    # Plot Frobenius
    for i in range(degree):
        axs[1].plot(x_range, frobs[:, i], label=f'Frobenius order {i}')
    axs[1].set_title("Algebraic Frobenius Orbit Features (x -> x^2^i)")
    axs[1].legend()
    axs[1].grid(True)

    # Plot Cyclotomic (Sin part)
    for i in range(degree):
        axs[2].plot(x_range, cycs[:, i*2], label=f'Cyclotomic sin(2pi*x/{i+1})')
    axs[2].set_title("Discrete Cyclotomic Features (Symmetry Orbits)")
    axs[2].legend()
    axs[2].grid(True)

    # Plot Reconstruction
    axs[3].scatter(x_range, y_noisy, color='red', alpha=0.3, label='Noisy Algebraic Signal')
    axs[3].plot(x_range, y_true, 'k--', label='True Signal')
    axs[3].plot(x_range, y_pred, 'g-', label='Algebraic Basis Reconstruction')
    axs[3].set_title("Signal Reconstruction using Purely Algebraic/Categorical Basis")
    axs[3].legend()
    axs[3].grid(True)

    plt.tight_layout()
    plt.savefig("polyfit/claudev2/algebraic_advanced_evaluation.png")
    print("Advanced algebraic evaluation results saved to polyfit/claudev2/algebraic_advanced_evaluation.png")

if __name__ == "__main__":
    main()
