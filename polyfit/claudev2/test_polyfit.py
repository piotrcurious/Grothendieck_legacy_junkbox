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

def run_fitter_test(x_data, y_data, degree, lambda_val=0.0):
    cpp_code = f"""
#include "polyfit/claudev2/src/arduino_polyfit.hpp"
#include "polyfit/claudev2/src/mock_arduino.hpp"

using namespace polyfit;

void setup() {{
    PolynomialFitter fitter({degree});
    float x_data[] = {{ {", ".join(map(str, x_data))} }};
    float y_data[] = {{ {", ".join(map(str, y_data))} }};
    size_t n = {len(x_data)};

    if (fitter.fit(x_data, y_data, n, {lambda_val}f)) {{
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

def main():
    # Original data
    x_data = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    y_data = np.array([5.0, 4.5, 3.0, 3.5, 9.0, 22.5])

    weights = run_fitter_test(x_data, y_data, 3)

    x_fine = np.linspace(0, 5, 100)
    y_fine = np.poly1d(weights[::-1])(x_fine)

    plt.figure(figsize=(10, 6))
    plt.scatter(x_data, y_data, color='red', label='Original Data')
    plt.plot(x_fine, y_fine, '-', color='green', label='C++ Fit (Degree 3)')
    plt.title("Polynomial Fitting: Arduino C++")
    plt.legend()
    plt.grid(True)
    plt.savefig("polyfit/claudev2/test_results.png")
    print("Test results saved to polyfit/claudev2/test_results.png")

if __name__ == "__main__":
    main()
