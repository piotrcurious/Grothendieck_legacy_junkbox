import numpy as np
import random
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial

# Generate sample data simulating irregular sampling intervals
def generate_sample_data(num_points=32):
    timestamps = np.sort(np.cumsum(np.random.randint(50, 500, size=num_points)))
    values = np.random.randint(0, 2, size=num_points)  # Binary values (0 or 1)
    return timestamps, values

# Approximate derivative using finite differences
def approximate_derivative(timestamps, values):
    derivatives = []
    for i in range(1, len(values)):
        delta_time = timestamps[i] - timestamps[i - 1]
        delta_value = values[i] - values[i - 1]
        derivative = delta_value / delta_time if delta_time != 0 else 0
        derivatives.append(derivative)
    return derivatives

# Generate polynomial candidates using least squares fitting
def generate_polynomial_candidates(timestamps, values, max_degree=5):
    candidates = []
    for degree in range(1, max_degree + 1):
        # Fit polynomial of current degree using least squares method
        coeffs = np.polyfit(timestamps, values, degree)
        poly = Polynomial(coeffs[::-1])
        candidates.append(poly)
    return candidates

# Evaluate polynomials by computing the sum of squared errors
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

# Main process
timestamps, values = generate_sample_data()
derivatives = approximate_derivative(timestamps, values)
candidates = generate_polynomial_candidates(timestamps, values)

best_poly, best_error = find_best_polynomial(candidates, timestamps, values)

# Print and visualize results
print("Best Polynomial Coefficients:", best_poly.coef)
print("Best Polynomial Error:", best_error)

# Plotting the sample data and best fit polynomial
plt.scatter(timestamps, values, color='red', label='Sample Data')
plt.plot(timestamps, best_poly(timestamps), label='Best Fit Polynomial', color='blue')
plt.xlabel('Timestamps')
plt.ylabel('Values')
plt.title('Sample Data and Best Fit Polynomial')
plt.legend()
plt.show()
