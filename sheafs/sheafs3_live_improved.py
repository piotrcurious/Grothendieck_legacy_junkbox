import serial
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial
import argparse
import sys
from sheaf_utils import MockSerial, parse_data_line

def parse_args():
    parser = argparse.ArgumentParser(description='Live visualization of sheaf data from Arduino.')
    parser.add_argument('--port', type=str, default='COM3', help='Serial port (e.g., COM3 or /dev/ttyACM0). Use "mock" for testing.')
    parser.add_argument('--baud', type=int, default=9600, help='Baud rate.')
    return parser.parse_args()

# Function to generate polynomial candidates using least squares fitting
def generate_polynomial_candidates(timestamps, values, max_degree=5):
    candidates = []
    # Degree cannot exceed number of points - 1
    actual_max_degree = min(max_degree, len(timestamps) - 1)
    for degree in range(1, actual_max_degree + 1):
        # Fit polynomial of current degree using least squares method
        coeffs = np.polyfit(timestamps, values, degree)
        poly = Polynomial(coeffs[::-1])
        candidates.append(poly)
    return candidates

# Function to evaluate polynomials by computing the sum of squared errors
def evaluate_polynomial(poly, timestamps, values):
    predicted_values = poly(timestamps)
    error = np.sum((predicted_values - values) ** 2)
    return error

# Find the best polynomial match
def find_best_polynomial(candidates, timestamps, values):
    best_poly = None
    best_error = float('inf')
    for poly in candidates:
        error = evaluate_polynomial(poly, timestamps, values)
        if error < best_error:
            best_error = error
            best_poly = poly
    return best_poly, best_error

# Plotting function
def plot_data_and_fit(timestamps, values, best_poly):
    plt.scatter(timestamps, values, color='red', label='Sample Data')
    if best_poly:
        # Generate more points for a smoother line
        if len(timestamps) > 1:
            t_smooth = np.linspace(min(timestamps), max(timestamps), 100)
            plt.plot(t_smooth, best_poly(t_smooth), label='Best Fit Polynomial', color='blue')
    plt.xlabel('Timestamps (ms)')
    plt.ylabel('Values')
    plt.title('Sample Data and Best Fit Polynomial')
    plt.legend()

def main():
    args = parse_args()
    print(f"Starting with port={args.port}")

    if args.port == 'mock':
        ser = MockSerial()
        print("Using mock serial source...")
    else:
        try:
            ser = serial.Serial(args.port, args.baud)
            ser.flushInput()
        except Exception as e:
            print(f"Error opening serial port {args.port}: {e}")
            sys.exit(1)

    timestamps = []
    values = []
    line = ""

    try:
        plt.ion()
        while True:
            if ser.in_waiting > 0:
                raw_line = ser.readline()
                line = raw_line.decode('utf-8', errors='replace').strip()
                if not line: continue
                print(line)
                sys.stdout.flush()

                timestamp, value = parse_data_line(line)
                if timestamp is not None and value is not None:
                    timestamps.append(timestamp)
                    values.append(value)

                    if len(timestamps) > 2:
                        candidates = generate_polynomial_candidates(timestamps, values)
                        best_poly, best_error = find_best_polynomial(candidates, timestamps, values)

                        print(f"Best Polynomial Coefficients: {best_poly.coef}")
                        print(f"Best Polynomial Error: {best_error}")
                        sys.stdout.flush()

                        plt.clf()
                        plot_data_and_fit(timestamps, values, best_poly)
                        plt.draw()
                        plt.pause(0.01)

            if "Data Collection Complete" in line:
                print("Cycle complete. Clearing data for next cycle.")
                sys.stdout.flush()
                timestamps = []
                values = []
                line = ""

    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        ser.close()

if __name__ == "__main__":
    main()
