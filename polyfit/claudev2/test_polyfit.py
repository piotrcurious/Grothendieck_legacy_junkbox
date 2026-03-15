import subprocess
import os
import numpy as np
import matplotlib.pyplot as plt

def compile_and_run_cpp():
    # Paths
    src_dir = "polyfit/claudev2/src"
    example_file = "polyfit/claudev2/example.ino"

    # Create a temporary main.cpp that includes example.ino content
    with open(example_file, 'r') as f:
        ino_content = f.read()

    # Remove #include "src/arduino_polyfit.hpp" as we'll compile from root of claudev2
    # and #include "src/mock_arduino.hpp"
    # Actually, it's easier to just use the existing main.cpp and link everything.

    # We need to compile:
    # src/arduino_polyfit.cpp
    # src/mock_arduino.cpp
    # src/main.cpp
    # AND the content of example.ino (which we can rename to example.cpp)

    with open("polyfit/claudev2/example.cpp", 'w') as f:
        f.write(ino_content.replace('src/', 'polyfit/claudev2/src/'))

    cmd = [
        "g++",
        "-I", ".",
        "polyfit/claudev2/src/arduino_polyfit.cpp",
        "polyfit/claudev2/src/mock_arduino.cpp",
        "polyfit/claudev2/src/main.cpp",
        "polyfit/claudev2/example.cpp",
        "-o", "polyfit/claudev2/test_bin"
    ]

    print(f"Compiling: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Compilation failed!")
        print(result.stderr)
        return None

    print("Running binary...")
    result = subprocess.run(["./polyfit/claudev2/test_bin"], capture_output=True, text=True)
    return result.stdout

def parse_output(output):
    lines = output.splitlines()
    predictions = []
    weights = []
    for line in lines:
        if "Weights:" in line:
            weights = [float(w) for w in line.split("Weights:")[1].strip().split()]
        if "x =" in line and "-> y =" in line:
            parts = line.split("=")
            x = float(parts[1].split("->")[0].strip())
            y = float(parts[2].strip())
            predictions.append((x, y))
    return weights, predictions

def test_and_plot():
    output = compile_and_run_cpp()
    if output is None:
        return

    print("Output from C++:")
    print(output)

    weights, predictions = parse_output(output)

    # Original data from example.ino
    x_data = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    y_data = np.array([5.0, 4.5, 3.0, 3.5, 9.0, 22.5])

    # C++ predictions
    test_x_cpp = np.array([p[0] for p in predictions])
    test_y_cpp = np.array([p[1] for p in predictions])

    # Numpy fit for comparison
    poly_coeffs = np.polyfit(x_data, y_data, 3)
    p = np.poly1d(poly_coeffs)

    x_fine = np.linspace(0, 5, 100)
    y_fine_numpy = p(x_fine)
    y_fine_cpp = np.poly1d(weights[::-1])(x_fine)

    plt.figure(figsize=(10, 6))
    plt.scatter(x_data, y_data, color='red', label='Original Data')
    plt.plot(x_fine, y_fine_numpy, '--', color='blue', label='Numpy Fit (Degree 3)')
    plt.plot(x_fine, y_fine_cpp, '-', color='green', alpha=0.6, label='Arduino C++ Fit (Degree 3)')
    plt.scatter(test_x_cpp, test_y_cpp, color='black', marker='x', label='C++ Predictions')

    plt.title("Polynomial Fitting: Arduino C++ vs Numpy")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.savefig("polyfit/claudev2/test_results.png")
    print("Test results saved to polyfit/claudev2/test_results.png")

if __name__ == "__main__":
    test_and_plot()
