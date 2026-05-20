import serial
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial
import argparse
import sys
from sheaf_utils import MockSerial, parse_data_line

# Simplified Feature Extraction instead of sklearn FeatureHasher
def extract_features_simple(timestamps, values, n_features=10):
    # A simple hash-based feature extraction
    features = np.zeros(n_features)
    for i in range(len(timestamps)):
        # Feature 1: Time diff
        dt = timestamps[i] - timestamps[i-1] if i > 0 else 0
        idx_dt = hash(f"dt_{dt}") % n_features
        features[idx_dt] += 1

        # Feature 2: Value
        val = values[i]
        idx_val = hash(f"val_{val}") % n_features
        features[idx_val] += 1

    norm = np.linalg.norm(features)
    if norm > 0:
        features /= norm
    return features

# Simplified Smoothing instead of scipy gaussian_filter1d
def simple_moving_average(values, window=3):
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window)/window, mode='same')

# Function to generate polynomial candidates guided by hashed features
def generate_polynomial_candidates(timestamps, values, max_degree=5):
    candidates = []
    smoothed_values = simple_moving_average(values)
    actual_max_degree = min(max_degree, len(timestamps) - 1)
    for degree in range(1, actual_max_degree + 1):
        coeffs = np.polyfit(timestamps, smoothed_values, degree)
        poly = Polynomial(coeffs[::-1])
        candidates.append(poly)
    return candidates

# Function to evaluate polynomials using feature-guided correlation
def evaluate_polynomial(poly, timestamps, values, target_features):
    predicted_values = poly(timestamps)
    error = np.sum((predicted_values - values) ** 2)

    poly_features = extract_features_simple(timestamps, predicted_values)
    feature_correlation = np.dot(target_features, poly_features)

    return error, feature_correlation

# Find the best polynomial match from candidates using feature correlation
def find_best_polynomial(candidates, timestamps, values, target_features):
    best_poly = None
    best_error = float('inf')
    best_correlation = -1.0
    for poly in candidates:
        error, correlation = evaluate_polynomial(poly, timestamps, values, target_features)
        # Prioritize lower error, then higher correlation
        if error < best_error:
            best_error = error
            best_poly = poly
            best_correlation = correlation
        elif error == best_error and correlation > best_correlation:
            best_poly = poly
            best_correlation = correlation
    return best_poly, best_error, best_correlation

def main():
    parser = argparse.ArgumentParser(description='Portable Sheaf Visualizer')
    parser.add_argument('--port', type=str, default='mock', help='Serial port or "mock"')
    args = parser.parse_args()

    if args.port == 'mock':
        ser = MockSerial()
    else:
        ser = serial.Serial(args.port, 9600)
        ser.flushInput()

    timestamps = []
    values = []
    sheafs = []
    errors = []

    SHEAF_TIME_THRESHOLD = 500
    SHEAF_SIZE_THRESHOLD = 5

    plt.ion()
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    line_buf = ""

    print(f"Starting {__file__} on port {args.port}")
    try:
        while True:
            if ser.in_waiting > 0:
                raw = ser.readline()
                line_buf = raw.decode('utf-8', errors='replace').strip()
                if not line_buf: continue

                timestamp, value = parse_data_line(line_buf)
                if timestamp is not None and value is not None:
                    timestamps.append(timestamp)
                    values.append(value)

                    # Update sheafs
                    if not sheafs or (timestamp - sheafs[-1][-1][0] > SHEAF_TIME_THRESHOLD):
                        sheafs.append([])
                    sheafs[-1].append((timestamp, value))

                    if len(timestamps) > SHEAF_SIZE_THRESHOLD:
                        target_features = extract_features_simple(timestamps, values)
                        candidates = generate_polynomial_candidates(timestamps, values)
                        best_poly, best_err, _ = find_best_polynomial(candidates, timestamps, values, target_features)
                        errors.append(best_err)

                        # Plot
                        axs[0].clear()
                        axs[0].scatter(timestamps, values, color='red', label='Data')
                        t_plot = np.linspace(min(timestamps), max(timestamps), 100)
                        axs[0].plot(t_plot, best_poly(t_plot), label=f'Fit (Err: {best_err:.2f})', color='blue')
                        axs[0].legend()

                        axs[1].clear()
                        axs[1].plot(errors, label='Error Trend')
                        axs[1].legend()

                        plt.draw()
                        plt.pause(0.01)

                if "Data Collection Complete" in line_buf:
                    print("Cycle complete.")
                    timestamps, values, sheafs, errors = [], [], [], []

    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        ser.close()

if __name__ == "__main__":
    main()
