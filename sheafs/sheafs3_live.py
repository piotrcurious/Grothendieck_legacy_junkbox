import serial
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial

# Set up the serial connection (adjust COM port and baud rate as needed)
ser = serial.Serial('COM3', 9600)  # Replace 'COM3' with your Arduino port
ser.flushInput()

# Arrays to hold the received data
timestamps = []
values = []

# Function to generate polynomial candidates using least squares fitting
def generate_polynomial_candidates(timestamps, values, max_degree=5):
    candidates = []
    for degree in range(1, max_degree + 1):
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
        plt.plot(timestamps, best_poly(timestamps), label='Best Fit Polynomial', color='blue')
    plt.xlabel('Timestamps')
    plt.ylabel('Values')
    plt.title('Sample Data and Best Fit Polynomial')
    plt.legend()
    plt.pause(0.05)  # Pause briefly to update the plot

# Parse line of data from Arduino
def parse_data_line(line):
    # Attempt to extract timestamp and value from the formatted string
    if "Collected Data Point" in line:
        try:
            parts = line.split(',')
            time_part = parts[0].split('Time: ')[1].strip().replace(' ms', '')
            value_part = parts[1].split('Value: ')[1].strip()
            timestamp = int(time_part)
            value = int(value_part)
            return timestamp, value
        except (IndexError, ValueError):
            print(f"Failed to parse data line: {line}")
    return None, None

# Main loop to read data from the serial port and process it
try:
    plt.ion()  # Turn on interactive mode for live updating of plots
    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()  # Read line from Arduino
            print(line)  # Display the raw line for debugging
            
            # Parse data if it's a collected data point
            timestamp, value = parse_data_line(line)
            if timestamp is not None and value is not None:
                timestamps.append(timestamp)
                values.append(value)

                # Ensure we have enough data points to fit polynomials
                if len(timestamps) > 2:
                    # Generate and evaluate polynomial candidates
                    candidates = generate_polynomial_candidates(timestamps, values)
                    best_poly, best_error = find_best_polynomial(candidates, timestamps, values)
                    
                    # Print best polynomial info
                    print(f"Best Polynomial Coefficients: {best_poly.coef}")
                    print(f"Best Polynomial Error: {best_error}")

                    # Update the plot with the latest data and best fit
                    plt.clf()  # Clear the previous plot
                    plot_data_and_fit(timestamps, values, best_poly)

except KeyboardInterrupt:
    print("Exiting...")
finally:
    ser.close()
