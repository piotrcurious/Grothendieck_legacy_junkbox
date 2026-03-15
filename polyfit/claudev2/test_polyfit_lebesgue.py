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

def run_lebesgue_test(x_data, y_data, degree):
    cpp_code = f"""
#include "polyfit/claudev2/src/arduino_polyfit.hpp"
#include "polyfit/claudev2/src/mock_arduino.hpp"

using namespace polyfit;

void setup() {{
    PolynomialFitter fitter({degree});
    float x_data[] = {{ {", ".join(map(str, x_data))} }};
    float y_data[] = {{ {", ".join(map(str, y_data))} }};
    size_t n = {len(x_data)};

    if (fitter.fit_lebesgue(x_data, y_data, n)) {{
        Serial.print("Weights: ");
        for (int i = 0; i <= fitter.degree; ++i) {{
            Serial.print(fitter.weights[i]);
            Serial.print(" ");
        }}
        Serial.println("");
    }} else {{
        Serial.println("Fit failed!");
    }}
}}

void loop() {{}}
"""
    with open("polyfit/claudev2/test_case.cpp", 'w') as f:
        f.write(cpp_code)

    if not compile_cpp():
        return None

    result = subprocess.run(["./polyfit/claudev2/test_bin"], capture_output=True, text=True)
    weights = []
    for line in result.stdout.splitlines():
        if "Weights:" in line:
            weights = [float(w) for w in line.split("Weights:")[1].strip().split()]
    return weights

def legendre_eval(n, x):
    if n == 0: return 1.0
    if n == 1: return x
    p0, p1 = 1.0, x
    for i in range(2, n + 1):
        p2 = ((2.0 * i - 1.0) * x * p1 - (i - 1.0) * p0) / i
        p0, p1 = p1, p2
    return p1

def predict_lebesgue(x, weights, x_min, x_max):
    x_norm = 2.0 * (x - x_min) / (x_max - x_min) - 1.0
    res = 0
    for d, w in enumerate(weights):
        res += w * legendre_eval(d, x_norm)
    return res

def main():
    # Test function: complex signal
    x = np.linspace(0, 10, 50)
    y = np.sin(x) + 0.5 * np.cos(2*x) + np.random.normal(0, 0.1, size=x.shape)

    degree = 6
    weights = run_lebesgue_test(x, y, degree)

    x_fine = np.linspace(0, 10, 200)
    y_pred = [predict_lebesgue(xi, weights, x.min(), x.max()) for xi in x_fine]

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='red', label='Data')
    plt.plot(x_fine, y_pred, 'g-', label=f'Lebesgue Fit (Legendre deg {degree})')
    plt.title("Lebesgue-based Orthogonal Projection Fit")
    plt.legend()
    plt.grid(True)
    plt.savefig("polyfit/claudev2/lebesgue_test_results.png")
    print("Lebesgue test results saved to polyfit/claudev2/lebesgue_test_results.png")

if __name__ == "__main__":
    main()
