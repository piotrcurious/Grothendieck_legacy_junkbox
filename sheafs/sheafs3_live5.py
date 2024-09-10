import serial
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial
from scipy.ndimage import gaussian_filter1d
from sklearn.feature_extraction import FeatureHasher

# Set up the serial connection (adjust COM port and baud rate as needed)
ser = serial.Serial('COM3', 9600)  # Replace 'COM3' with your Arduino port
ser.flushInput()

# Variables to hold the received data
timestamps = []
values = []
sheafs = []  # List to hold sheafs of data points
errors = []  # List to track error metrics over time

# Sheaf parameters
SHEAF_TIME_THRESHOLD = 500  # Initial time threshold (in ms)
SHEAF_SIZE_THRESHOLD = 5  # Minimum size of a sheaf to be considered for fitting

# Feature hasher settings
HASH_DIMENSIONS = 10  # Dimensionality of the hashed feature space

# Initialize the feature hasher
hasher = FeatureHasher(n_features=HASH_DIMENSIONS, input_type='string')

# Function to extract features from data using feature hashing
def extract_features(timestamps, values):
    features = []
    for i in range(len(timestamps)):
        features.append(f'time_diff_{i}:{timestamps[i] - timestamps[i - 1] if i > 0 else 0}')
        features.append(f'value_{i}:{values[i]}')
    # Hash the features into a fixed-size vector
    hashed_features = hasher.transform([features]).toarray().flatten()
    return hashed_features

# Function to generate polynomial candidates guided by hashed features
def generate_polynomial_candidates(timestamps, values, hashed_features):
    max_degree = min(len(hashed_features), 5)  # Limit degree based on hashed feature dimensions
    candidates = []
    smoothed_values = gaussian_filter1d(values, sigma=1)
    for degree in range(1, max_degree + 1):
        coeffs = np.polyfit(timestamps, smoothed_values, degree)
        poly = Polynomial(coeffs[::-1])
        candidates.append(poly)
    return candidates

# Function to evaluate polynomials using feature-guided hashing comparison
def evaluate_polynomial(poly, timestamps, values, hashed_features):
    predicted_values = poly(timestamps)
    error = np.sum((predicted_values - values) ** 2)
    # Compare polynomial fit against hashed features by correlation
    poly_features = extract_features(timestamps, predicted_values)
    feature_correlation = np.dot(hashed_features, poly_features) / (
        np.linalg.norm(hashed_features) * np.linalg.norm(poly_features) + 1e-8
    )
    return error, feature_correlation

# Find the best polynomial match from candidates using feature correlation
def find_best_polynomial(candidates, timestamps, values, hashed_features):
    best_poly = None
    best_error = float('inf')
    best_correlation = 0
    for poly in candidates:
        error, correlation = evaluate_polynomial(poly, timestamps, values, hashed_features)
        # Optimize for low error and high feature correlation
        if error < best_error and correlation > best_correlation:
            best_error = error
            best_poly = poly
            best_correlation = correlation
    return best_poly, best_error, best_correlation

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

# Function to dynamically update sheafs based on data patterns
def update_sheafs(timestamps, values):
    if not timestamps or not values:
        return

    # Check if we should start a new sheaf based on adaptive criteria
    if not sheafs or (timestamps[-1] - sheafs[-1][-1][0] > SHEAF_TIME_THRESHOLD) or abs(values[-1] - sheafs[-1][-1][1]) > 50:
        sheafs.append([])

    # Add the current data point to the latest sheaf
    sheafs[-1].append((timestamps[-1], values[-1]))

    # Dynamically adjust the sheaf threshold based on recent time differences
    if len(sheafs[-1]) > 1:
        recent_timestamps = [point[0] for point in sheafs[-1][-5:]]
        if len(recent_timestamps) > 1:
            time_diffs = np.diff(recent_timestamps)
            avg_diff = np.mean(time_diffs)
            global SHEAF_TIME_THRESHOLD
            SHEAF_TIME_THRESHOLD = max(250, min(avg_diff * 2, 1000))

# Function to aggregate polynomials from all sheafs and improve fitting
def aggregate_polynomials(sheafs):
    combined_timestamps = []
    combined_values = []
    for sheaf in sheafs:
        sheaf_timestamps = [point[0] for point in sheaf]
        sheaf_values = [point[1] for point in sheaf]
        combined_timestamps.extend(sheaf_timestamps)
        combined_values.extend(sheaf_values)

    if len(combined_timestamps) > SHEAF_SIZE_THRESHOLD:
        hashed_features = extract_features(combined_timestamps, combined_values)
        candidates = generate_polynomial_candidates(combined_timestamps, combined_values, hashed_features)
        best_poly, best_error, best_correlation = find_best_polynomial(candidates, combined_timestamps, combined_values, hashed_features)
        errors.append(best_error)  # Track the best error
        return best_poly, best_error, best_correlation
    return None, None, None

# Plotting function to visualize data, sheafs, and polynomial fit with error tracking
def plot_data_and_fit(timestamps, values, best_poly, sheafs, best_error):
    plt.clf()  # Clear the plot
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Plot sample data and best fit polynomial
    axs[0].scatter(timestamps, values, color='red', label='Sample Data')
    if best_poly:
        axs[0].plot(timestamps, best_poly(timestamps), label=f'Best Fit Polynomial (Error: {best_error:.2f})', color='blue')
    
    # Highlight sheafs with different colors
    colors = ['orange', 'green', 'purple', 'cyan']
    for i, sheaf in enumerate(sheafs):
        sheaf_timestamps = [point[0] for point in sheaf]
        sheaf_values = [point[1] for point in sheaf]
        axs[0].scatter(sheaf_timestamps, sheaf_values, color=colors[i % len(colors)], label=f'Sheaf {i + 1}')

    axs[0].set_xlabel('Timestamps')
    axs[0].set_ylabel('Values')
    axs[0].set_title('Sample Data and Best Fit Polynomial')
    axs[0].legend()

    # Plot error evolution over time
    axs[1].plot(errors, color='blue', label='Fit Error Over Time')
    axs[1].set_xlabel('Iterations')
    axs[1].set_ylabel('Error')
    axs[1].set_title('Polynomial Fit Error Over Time')
    axs[1].legend()

    plt.tight_layout()
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
                best_poly, best_error, best_correlation = aggregate_polynomials(sheafs)

                # Update the plot with the latest data, sheafs, and best fit
                plot_data_and_fit(timestamps, values, best_poly, sheafs, best_error)

except KeyboardInterrupt:
    print("Exiting...")
finally:
    ser.close()
