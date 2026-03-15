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
    x_comp = np.linspace(0, 2*np.pi, 30)
    y_comp = np.sin(x_comp) + np.sin(2*x_comp)
    x_comp_test = np.linspace(0, 2*np.pi, 100)
    y_comp_test = np.sin(x_comp_test) + np.sin(2*x_comp_test)

    # Scenario 2: Decaying Sine
    x_decay = np.linspace(0, 10, 50)
    y_decay = np.exp(-0.2*x_decay) * (np.sin(x_decay) + np.sin(2*x_decay))
    x_decay_test = np.linspace(0, 10, 100)
    y_decay_test = np.exp(-0.2*x_decay_test) * (np.sin(x_decay_test) + np.sin(2*x_decay_test))

    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    # Run Scenario 1
    w1 = run_fitter_test(x_comp, y_comp, 7, lambda_val=0.001)
    plot_scenario(axs[0], x_comp, y_comp, x_comp_test, y_comp_test, w1, "Compound Sine", 7)

    # Run Scenario 2
    w2 = run_fitter_test(x_decay, y_decay, 8, lambda_val=0.01)
    plot_scenario(axs[1], x_decay, y_decay, x_decay_test, y_decay_test, w2, "Decaying Sine", 8)

    plt.tight_layout()
    plt.savefig("polyfit/claudev2/complex_test_results.png")
    print("Complex test results saved to polyfit/claudev2/complex_test_results.png")

if __name__ == "__main__":
    main()
