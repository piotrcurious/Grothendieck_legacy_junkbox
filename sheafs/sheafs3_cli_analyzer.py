import numpy as np
from numpy.polynomial import Polynomial
import argparse
import sys
from sheaf_utils import MockSerial, parse_data_line

def parse_args():
    parser = argparse.ArgumentParser(description='Simplified sheaf data analyzer (Headless).')
    parser.add_argument('--port', type=str, default='mock', help='Mock data source or serial port.')
    parser.add_argument('--max-samples', type=int, default=32, help='Max samples to process.')
    return parser.parse_args()

def generate_polynomial_candidates(timestamps, values, max_degree=5):
    candidates = []
    actual_max_degree = min(max_degree, len(timestamps) - 1)
    for degree in range(1, actual_max_degree + 1):
        coeffs = np.polyfit(timestamps, values, degree)
        poly = Polynomial(coeffs[::-1])
        candidates.append(poly)
    return candidates

def evaluate_polynomial(poly, timestamps, values):
    predicted_values = poly(timestamps)
    error = np.sum((predicted_values - values) ** 2)
    return error

def find_best_polynomial(candidates, timestamps, values):
    best_poly = None
    best_error = float('inf')
    for poly in candidates:
        error = evaluate_polynomial(poly, timestamps, values)
        if error < best_error:
            best_error = error
            best_poly = poly
    return best_poly, best_error

def main():
    args = parse_args()

    if args.port == 'mock':
        ser = MockSerial(use_lfsr=True)
    else:
        import serial
        ser = serial.Serial(args.port, 9600)
        ser.flushInput()

    timestamps = []
    values = []

    print(f"Starting CLI analyzer on port {args.port}...")
    try:
        while len(timestamps) < args.max_samples:
            raw = ser.readline()
            line = raw.decode('utf-8', errors='replace').strip()
            if not line: continue

            timestamp, value = parse_data_line(line)
            if timestamp is not None and value is not None:
                timestamps.append(timestamp)
                values.append(value)

                if len(timestamps) > 2:
                    candidates = generate_polynomial_candidates(timestamps, values)
                    best_poly, best_error = find_best_polynomial(candidates, timestamps, values)
                    print(f"Sample {len(timestamps)}: Best Degree: {len(best_poly.coef)-1}, Error: {best_error:.4f}")

            if "Data Collection Complete" in line:
                if len(timestamps) >= args.max_samples: break
                print("Cycle complete. Continuing...")

        print("Batch complete.")
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        ser.close()

if __name__ == "__main__":
    main()
