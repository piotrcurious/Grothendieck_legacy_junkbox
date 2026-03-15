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

def run_evaluation_test(x_data, y_data, d_base, d_res):
    cpp_code = f"""
#include "polyfit/claudev2/src/arduino_polyfit.hpp"
#include "polyfit/claudev2/src/mock_arduino.hpp"

using namespace polyfit;

void setup() {{
    float x_data[] = {{ {", ".join(map(str, x_data))} }};
    float y_data[] = {{ {", ".join(map(str, y_data))} }};
    size_t n = {len(x_data)};

    // 1. Single layer
    PolynomialFitter single_layer({d_base});
    single_layer.fit(x_data, y_data, n);

    // 2. Residual Layer
    ResidualFitter multi_layer({d_base}, {d_res});
    multi_layer.fit(x_data, y_data, n);

    Serial.println("--- Evaluation Results ---");
    Serial.print("Weights Single: ");
    for(int i=0; i<={d_base}; ++i) {{ Serial.print(single_layer.weights[i]); Serial.print(" "); }} Serial.println("");

    Serial.print("Weights Multi Base: ");
    for(int i=0; i<={d_base}; ++i) {{ Serial.print(multi_layer.base_fitter.weights[i]); Serial.print(" "); }} Serial.println("");

    Serial.print("Weights Multi Res: ");
    for(int i=0; i<={d_res}; ++i) {{ Serial.print(multi_layer.residual_fitter.weights[i]); Serial.print(" "); }} Serial.println("");
}}

void loop() {{}}
"""
    with open("polyfit/claudev2/test_case.cpp", 'w') as f:
        f.write(cpp_code)

    if not compile_cpp():
        return None

    result = subprocess.run(["./polyfit/claudev2/test_bin"], capture_output=True, text=True)

    weights = {}
    for line in result.stdout.splitlines():
        if "Weights Single:" in line:
            weights['single'] = [float(w) for w in line.split("Weights Single:")[1].strip().split()]
        if "Weights Multi Base:" in line:
            weights['multi_base'] = [float(w) for w in line.split("Weights Multi Base:")[1].strip().split()]
        if "Weights Multi Res:" in line:
            weights['multi_res'] = [float(w) for w in line.split("Weights Multi Res:")[1].strip().split()]
    return weights

def main():
    np.random.seed(42)
    # Signal: Sine wave + Noise + Cubic trend
    x = np.linspace(0, 5, 40)
    y_true_func = lambda t: 0.2 * t**3 + np.sin(2 * t)
    y = y_true_func(x) + np.random.normal(0, 0.2, size=x.shape)

    d_base = 2
    d_res = 4

    weights = run_evaluation_test(x, y, d_base, d_res)

    x_fine = np.linspace(0, 5, 100)
    y_true_fine = y_true_func(x_fine)
    y_single = np.poly1d(weights['single'][::-1])(x_fine)
    y_multi_base = np.poly1d(weights['multi_base'][::-1])(x_fine)
    y_multi_res = np.poly1d(weights['multi_res'][::-1])(x_fine)
    y_multi = y_multi_base + y_multi_res

    plt.figure(figsize=(12, 8))
    plt.scatter(x, y, color='red', alpha=0.5, label='Noisy Data')
    plt.plot(x_fine, y_true_fine, 'k--', label='True Signal')
    plt.plot(x_fine, y_single, 'b-', label=f'Single Layer (deg {d_base})')
    plt.plot(x_fine, y_multi, 'g-', label=f'Multi-Layer Residual (deg {d_base}+{d_res})')

    plt.title("Categorical Paradigm Shift: Multi-Layer Residual Fitting")
    plt.legend()
    plt.grid(True)
    plt.savefig("polyfit/claudev2/paradigm_evaluation_results.png")
    print("Paradigm evaluation results saved to polyfit/claudev2/paradigm_evaluation_results.png")

if __name__ == "__main__":
    main()
