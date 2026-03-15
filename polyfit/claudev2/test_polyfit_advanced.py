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

def run_test_case(x_data, y_data, degree, lambda_val=0.0):
    # Generate C++ test case file
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

    # Parse weights
    weights = []
    for line in result.stdout.splitlines():
        if "Weights:" in line:
            weights = [float(w) for w in line.split("Weights:")[1].strip().split()]
    return weights

def calculate_mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def plot_scenario(ax, x_train, y_train, x_test, y_test, weights, title, degree):
    poly = np.poly1d(weights[::-1])
    x_range = np.linspace(min(x_train.min(), x_test.min()), max(x_train.max(), x_test.max()), 100)

    ax.scatter(x_train, y_train, color='red', label='Train Data')
    ax.scatter(x_test, y_test, color='blue', alpha=0.5, label='Test Data')
    ax.plot(x_range, poly(x_range), 'g-', label=f'Fit (deg {degree})')

    mse_train = calculate_mse(y_train, poly(x_train))
    mse_test = calculate_mse(y_test, poly(x_test))

    ax.set_title(f"{title}\nMSE Train: {mse_train:.4f}, Test: {mse_test:.4f}")
    ax.legend()
    ax.grid(True)

def main():
    np.random.seed(42)

    # Scenario 1: Clean Cubic Data
    x_cubic = np.linspace(-2, 2, 20)
    y_cubic = 0.5 * x_cubic**3 - 2 * x_cubic**2 + x_cubic + 5

    # Scenario 2: Noisy Quadratic Data
    x_noisy = np.linspace(-3, 3, 30)
    y_noisy = -x_noisy**2 + 2 * x_noisy + 10 + np.random.normal(0, 1.0, size=x_noisy.shape)

    # Scenario 3: Overfitting (High degree, few points)
    x_overfit = np.linspace(0, 5, 6)
    y_overfit = np.sin(x_overfit) + np.random.normal(0, 0.1, size=x_overfit.shape)
    x_overfit_test = np.linspace(0, 5, 50)
    y_overfit_test = np.sin(x_overfit_test)

    fig, axs = plt.subplots(2, 2, figsize=(15, 12))

    # Run Scenario 1
    w1 = run_test_case(x_cubic, y_cubic, 3)
    plot_scenario(axs[0, 0], x_cubic, y_cubic, x_cubic, y_cubic, w1, "Clean Cubic", 3)

    # Run Scenario 2
    w2 = run_test_case(x_noisy, y_noisy, 2)
    plot_scenario(axs[0, 1], x_noisy, y_noisy, x_noisy, y_noisy, w2, "Noisy Quadratic", 2)

    # Run Scenario 3 - Overfit No Reg
    w3_no_reg = run_test_case(x_overfit, y_overfit, 5)
    plot_scenario(axs[1, 0], x_overfit, y_overfit, x_overfit_test, y_overfit_test, w3_no_reg, "Overfitting (deg 5, no reg)", 5)

    # Run Scenario 3 - Overfit With Reg
    w3_reg = run_test_case(x_overfit, y_overfit, 5, lambda_val=1.0)
    plot_scenario(axs[1, 1], x_overfit, y_overfit, x_overfit_test, y_overfit_test, w3_reg, "Regularized (deg 5, lambda=1.0)", 5)

    plt.tight_layout()
    plt.savefig("polyfit/claudev2/advanced_test_results.png")
    print("Advanced test results saved to polyfit/claudev2/advanced_test_results.png")

if __name__ == "__main__":
    main()
