import subprocess
import os
import matplotlib.pyplot as plt

def compile_and_run():
    print("Compiling C++ tests...")
    compile_cmd = ["g++", "-o", "kahan_EMA/test_main", "kahan_EMA/test_main.cpp", "kahan_EMA/mock_arduino.cpp", "-I", "kahan_EMA"]
    subprocess.run(compile_cmd, check=True)

    print("Running C++ tests...")
    result = subprocess.run(["./kahan_EMA/test_main"], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Errors:")
        print(result.stderr)

def run_precision_comparison():
    print("Extracting data from C++ test output...")
    result = subprocess.run(["./kahan_EMA/test_main"], capture_output=True, text=True)
    lines = result.stdout.splitlines()

    steps = []
    s_history = []
    k_history = []

    is_data = False
    for line in lines:
        if line == "DATA_START":
            is_data = True
            continue
        if line == "DATA_END":
            is_data = False
            continue
        if is_data:
            parts = line.split(',')
            if len(parts) == 3:
                steps.append(int(parts[0]))
                s_history.append(float(parts[1]))
                k_history.append(float(parts[2]))

    if not steps:
        print("No data extracted!")
        return

    import numpy as np
    baseline = 100000.0
    tiny_increment = 1.0

    plt.figure(figsize=(12, 10))

    plt.subplot(2, 1, 1)
    plt.plot(s_history, label='Standard EMA (float32)')
    plt.plot(k_history, label='Kahan EMA (float32)')
    plt.axhline(y=baseline + tiny_increment, color='r', linestyle='--', label='Target')
    plt.title('Numerical Stability: Standard vs Kahan EMA')
    plt.xlabel('Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    target = baseline + tiny_increment
    s_error = [abs(x - target) for x in s_history]
    k_error = [abs(x - target) for x in k_history]
    plt.plot(steps, s_error, label='Standard EMA Error')
    plt.plot(steps, k_error, label='Kahan EMA Error')
    plt.yscale('log')
    plt.title('Log Error over Time')
    plt.xlabel('Steps')
    plt.ylabel('Absolute Error')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('kahan_EMA/stability_comparison.png')
    print("Saved stability_comparison.png")

    print(f"Final Standard EMA Error: {s_error[-1]}")
    print(f"Final Kahan EMA Error: {k_error[-1]}")

def run_sine_analysis():
    print("Extracting sine wave data from C++ test output...")
    result = subprocess.run(["./kahan_EMA/test_main"], capture_output=True, text=True)
    lines = result.stdout.splitlines()

    steps = []
    clean = []
    noisy = []
    filtered = []

    is_data = False
    for line in lines:
        if line == "SINE_DATA_START":
            is_data = True
            continue
        if line == "SINE_DATA_END":
            is_data = False
            continue
        if is_data:
            parts = line.split(',')
            if len(parts) == 4:
                steps.append(int(parts[0]))
                clean.append(float(parts[1]))
                noisy.append(float(parts[2]))
                filtered.append(float(parts[3]))

    if not steps:
        print("No sine data extracted!")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(steps, noisy, label='Noisy Signal', alpha=0.3, color='gray')
    plt.plot(steps, clean, label='Clean Signal', color='black', linestyle='--')
    plt.plot(steps, filtered, label='Kahan EMA Filtered', color='blue')
    plt.title('Kahan EMA Smoothing Performance')
    plt.xlabel('Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig('kahan_EMA/sine_smoothing.png')
    print("Saved sine_smoothing.png")

def run_benchmarks():
    print("Extracting benchmark results from C++ test output...")
    result = subprocess.run(["./kahan_EMA/test_main"], capture_output=True, text=True)
    lines = result.stdout.splitlines()

    for line in lines:
        if line.startswith("PERF:"):
            print(line)

def run_alpha_comparison():
    print("Extracting alpha comparison data from C++ test output...")
    result = subprocess.run(["./kahan_EMA/test_main"], capture_output=True, text=True)
    lines = result.stdout.splitlines()

    alpha_data = {}
    current_alpha = None

    is_data = False
    for line in lines:
        if line == "ALPHA_COMP_START":
            is_data = True
            continue
        if line == "ALPHA_COMP_END":
            is_data = False
            continue
        if is_data:
            if line.startswith("ALPHA:"):
                current_alpha = float(line.split(':')[1])
                alpha_data[current_alpha] = ([], [])
            else:
                parts = line.split(',')
                if len(parts) == 2 and current_alpha is not None:
                    alpha_data[current_alpha][0].append(int(parts[0]))
                    alpha_data[current_alpha][1].append(float(parts[1]))

    if not alpha_data:
        print("No alpha comparison data extracted!")
        return

    plt.figure(figsize=(10, 6))
    for alpha, (steps, vals) in alpha_data.items():
        plt.plot(steps, vals, label=f'alpha={alpha}')
    plt.axhline(y=10.0, color='r', linestyle='--', label='Target')
    plt.title('Convergence Speed for Different Alpha Values')
    plt.xlabel('Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig('kahan_EMA/alpha_comparison.png')
    print("Saved alpha_comparison.png")

if __name__ == "__main__":
    compile_and_run()
    run_benchmarks()
    run_precision_comparison()
    run_sine_analysis()
    run_alpha_comparison()
