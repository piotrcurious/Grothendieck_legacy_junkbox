import serial
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial

# Set up the serial connection (adjust COM port and baud rate as needed)
ser = serial.Serial('COM3', 9600)  # Replace 'COM3' with your Arduino port
ser.flushInput()

# Variables to hold the received data
timestamps = []
values = []
sheafs = []  # List to hold sheafs of data points

# Sheaf parameters
SHEAF_TIME_THRESHOLD = 500  # Initial time threshold to determine when to start a new sheaf (in ms)
SHEAF_SIZE_THRESHOLD = 5  # Minimum size of a sheaf to be considered for fitting

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

# Find the best polynomial match for a set of candidates
def find_best_polynomial(candidates, timestamps, values):
    best_poly = None
    best_error = float('inf')
    for poly in candidates:
        error = evaluate_polynomial(poly, timestamps, values)
        if error < best_error:
            best_error = error
            best_poly = poly
    return best_poly, best_error

# Function to parse a line of data from the Arduino
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
            print(f"Failed to parse data line: {line}")
    return None, None

# Function to update sheafs based on adaptive criteria
def update_sheafs(timestamps, values):
    if not timestamps or not values:
        return

    # Check if a new sheaf should start based on time threshold or a significant value change
    if not sheafs or (timestamps[-1] - sheafs[-1][-1][0] > SHEAF_TIME_THRESHOLD):
        sheafs.append([])

    # Add the latest data point to the current sheaf
    sheafs[-1].append((timestamps[-1], values[-1]))

    # Adjust the sheaf's time threshold adaptively based on data characteristics
    if len(sheafs[-1]) > 1:
        recent_timestamps = [point[0] for point in sheafs[-1][-5:]]  # Look at the last 5 points in the sheaf
        if len(recent_timestamps) > 1:
            time_diffs = np.diff(recent_timestamps)
            avg_diff = np.mean(time_diffs)
            SHEAF_TIME_THRESHOLD = max(250, min(avg_diff * 2, 1000))  # Adjust within reasonable bounds

# Function to aggregate polynomials from all sheafs
def aggregate_polynomials(sheafs):
    combined_timestamps = []
    combined_values = []
    for sheaf in sheafs:
        sheaf_timestamps = [point[0] for point in sheaf]
        sheaf_values = [point[1] for point in sheaf]
        combined_timestamps.extend(sheaf_timestamps)
        combined_values.extend(sheaf_values)

    if len(combined_timestamps) > SHEAF_SIZE_THRESHOLD:
        candidates = generate_polynomial_candidates(combined_timestamps, combined_values)
        best_poly, best_error = find_best_polynomial(candidates, combined_timestamps, combined_values)
        return best_poly, best_error
    return None, None

# Plotting function to visualize the current data and polynomial fit
def plot_data_and_fit(timestamps, values, best_poly, sheafs, best_error):
    plt.clf()  # Clear the plot
    plt.scatter(timestamps, values, color='red', label='Sample Data')
    if best_poly:
        plt.plot(timestamps, best_poly(timestamps), label=f'Best Fit Polynomial (Error: {best_error:.2f})', color='blue')
    
    # Highlight sheafs with different colors
    colors = ['orange', 'green', 'purple', 'cyan']
    for i, sheaf in enumerate(sheafs):
        sheaf_timestamps = [point[0] for point in sheaf]
        sheaf_values = [point[1] for point in sheaf]
        plt.scatter(sheaf_timestamps, sheaf_values, color=colors[i % len(colors)], label=f'Sheaf {i + 1}')

    plt.xlabel('Timestamps')
    plt.ylabel('Values')
    plt.title('Sample Data and Best Fit Polynomial')
    plt.legend()
    plt.pause(0.05)  # Pause briefly to update the plot

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
                update_sheafs(timestamps, values)  # Group data points into sheafs

                # Aggregate polynomials from current sheafs
                best_poly, best_error = aggregate_polynomials(sheafs)

                # Update the plot with the latest data, sheafs, and best fit
                plot_data_and_fit(timestamps, values, best_poly, sheafs, best_error)

except KeyboardInterrupt:
    print("Exiting...")
finally:
    ser.close()
