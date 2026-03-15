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
    ax.scatter(x_test, y_test, color='blue', alpha=0.3, label='True Func')
    ax.plot(x_range, poly(x_range), 'g-', label=f'Fit (deg {degree})')

    mse_train = calculate_mse(y_train, poly(x_train))
    mse_test = calculate_mse(y_test, poly(x_test))

    ax.set_title(f"{title}\nMSE Train: {mse_train:.4f}, Test: {mse_test:.4f}")
    ax.legend()
    ax.grid(True)

def main():
    np.random.seed(42)

    # Scenario 1: Compound Sine
    # y = sin(x) + sin(2x)
    x_comp = np.linspace(0, 2*np.pi, 30)
    y_comp = np.sin(x_comp) + np.sin(2*x_comp)
    x_comp_test = np.linspace(0, 2*np.pi, 100)
    y_comp_test = np.sin(x_comp_test) + np.sin(2*x_comp_test)

    # Scenario 2: Decaying Sine
    # y = exp(-0.2x) * (sin(x) + sin(2x))
    x_decay = np.linspace(0, 10, 50)
    y_decay = np.exp(-0.2*x_decay) * (np.sin(x_decay) + np.sin(2*x_decay))
    x_decay_test = np.linspace(0, 10, 100)
    y_decay_test = np.exp(-0.2*x_decay_test) * (np.sin(x_decay_test) + np.sin(2*x_decay_test))

    # Scenario 3: Mixed Harmonic
    # y = cos(x) + 0.5*sin(3x) + 0.1*x^2
    x_mixed = np.linspace(-5, 5, 40)
    y_mixed = np.cos(x_mixed) + 0.5*np.sin(3*x_mixed) + 0.1*x_mixed**2
    x_mixed_test = np.linspace(-5, 5, 100)
    y_mixed_test = np.cos(x_mixed_test) + 0.5*np.sin(3*x_mixed_test) + 0.1*x_mixed_test**2

    fig, axs = plt.subplots(2, 2, figsize=(15, 12))

    # Run Scenario 1 - High degree to approximate sine
    w1 = run_test_case(x_comp, y_comp, 7, lambda_val=0.001)
    plot_scenario(axs[0, 0], x_comp, y_comp, x_comp_test, y_comp_test, w1, "Compound Sine", 7)

    # Run Scenario 2 - High degree for decay
    w2 = run_test_case(x_decay, y_decay, 8, lambda_val=0.01)
    plot_scenario(axs[0, 1], x_decay, y_decay, x_decay_test, y_decay_test, w2, "Decaying Sine", 8)

    # Run Scenario 3
    w3 = run_test_case(x_mixed, y_mixed, 6, lambda_val=0.001)
    plot_scenario(axs[1, 0], x_mixed, y_mixed, x_mixed_test, y_mixed_test, w3, "Mixed Harmonic", 6)

    # Scenario 4: Step function approximation
    x_step = np.linspace(-1, 1, 50)
    y_step = np.where(x_step > 0, 1.0, 0.0)
    w4 = run_test_case(x_step, y_step, 9, lambda_val=0.1)
    plot_scenario(axs[1, 1], x_step, y_step, x_step, y_step, w4, "Step Function Approx", 9)

    plt.tight_layout()
    plt.savefig("polyfit/claudev2/complex_test_results.png")
    print("Complex test results saved to polyfit/claudev2/complex_test_results.png")

if __name__ == "__main__":
    main()
