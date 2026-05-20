import numpy as np
from numpy.polynomial import Polynomial
import argparse
import sys
import time

def parse_args():
    parser = argparse.ArgumentParser(description='Simplified sheaf data analyzer without heavy dependencies.')
    parser.add_argument('--port', type=str, default='mock', help='Mock data source only in this simplified version.')
    return parser.parse_args()

class MockSource:
    def __init__(self):
        self.time = 0
    def get_line(self):
        import random
        time.sleep(0.1)
        self.time += random.randint(50, 500)
        value = random.randint(0, 1)
        return f"Collected Data Point: Time: {self.time} ms, Value: {value}"

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

def parse_data_line(line):
    if "Collected Data Point" in line:
        try:
            parts = line.split(',')
            time_part = parts[0].split('Time: ')[1].strip().replace(' ms', '')
            value_part = parts[1].split('Value: ')[1].strip()
            timestamp = int(time_part)
            value = int(value_part)
            return timestamp, value
        except (IndexError, ValueError):
            pass
    return None, None

def main():
    args = parse_args()
    source = MockSource()

    timestamps = []
    values = []

    print("Starting simplified analyzer (CLI only)...")
    try:
        for _ in range(50): # Run for 50 samples
            line = source.get_line()
            print(line)

            timestamp, value = parse_data_line(line)
            if timestamp is not None and value is not None:
                timestamps.append(timestamp)
                values.append(value)

                if len(timestamps) > 2:
                    candidates = generate_polynomial_candidates(timestamps, values)
                    best_poly, best_error = find_best_polynomial(candidates, timestamps, values)
                    print(f"Best Degree: {len(best_poly.coef)-1}, Error: {best_error:.4f}")

            if len(timestamps) >= 32:
                print("Batch complete.")
                break
    except KeyboardInterrupt:
        print("Exiting...")

if __name__ == "__main__":
    main()
