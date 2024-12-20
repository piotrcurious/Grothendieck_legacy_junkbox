import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.linalg import lstsq


# 1. Polynomial fitting function
def fit_polynomial_sheaf(timestamps, values, degree=3):
    """
    Fits a polynomial of a given degree using Grothendieck's sheaf approach.
    The polynomial represents a local section over the data.
    """
    # Solve using least squares for consistency
    A = np.vander(timestamps, degree + 1, increasing=False)
    coeffs, _, _, _ = lstsq(A, values)
    poly = np.poly1d(coeffs[::-1])  # Flip coefficients for np.poly1d
    return poly, coeffs


# 2. Merging local sections into a global polynomial
def merge_polynomials(local_sections):
    """
    Merges local polynomial sections into a global polynomial by ensuring consistency.
    """
    merged_coeffs = np.zeros_like(local_sections[0][1])  # Initialize coefficients
    weight = np.zeros_like(merged_coeffs)  # For weighted averaging

    for poly, coeffs, window_t in local_sections:
        merged_coeffs += coeffs * len(window_t)  # Weight by section size
        weight += len(window_t)

    return np.poly1d(merged_coeffs / weight)  # Average the coefficients


# 3. Extracting Galois-inspired features
def extract_features_galois(poly):
    """
    Extracts roots (splitting fields) and critical points (transitions) of the polynomial.
    """
    x = sp.Symbol('x')
    poly_expr = sp.Poly.from_list(poly.coefficients[::-1], x)

    # Roots as splitting fields
    roots = sp.solvers.solve(poly_expr, x)

    # Critical points (first derivative = 0)
    deriv = sp.diff(poly_expr.as_expr(), x)
    critical_points = sp.solvers.solve(deriv, x)

    return roots, critical_points


# 4. Local polynomial fitting using sheaves
def local_polynomial_fitting_sheaf(timestamps, values, window_size=10, degree=3):
    """
    Fits local polynomial sections to the dataset using sliding windows.
    """
    local_sections = []
    num_points = len(timestamps)

    for i in range(0, num_points, window_size):
        # Extract the window
        window_t = timestamps[i:i + window_size]
        window_v = values[i:i + window_size]

        if len(window_t) >= degree + 1:  # Ensure sufficient data points
            poly, coeffs = fit_polynomial_sheaf(window_t, window_v, degree)
            local_sections.append((poly, coeffs, window_t))

    return local_sections


# 5. Visualization of global and local fits
def plot_sheaf_results(timestamps, values, global_poly, local_sections, roots, critical_points):
    """
    Plots the dataset, global fit, local fits, roots, and critical points.
    """
    plt.figure(figsize=(12, 8))
    
    # Original data
    plt.scatter(timestamps, values, label="Original Data", alpha=0.6)
    
    # Global fit
    plt.plot(timestamps, global_poly(timestamps), label="Global Fit", color="red", linewidth=2)
    
    # Local fits
    for poly, _, window_t in local_sections:
        t_fit = np.linspace(min(window_t), max(window_t), 100)
        plt.plot(t_fit, poly(t_fit), '--', label=f"Local Fit [{int(min(window_t))}-{int(max(window_t))}]")
    
    # Roots
    roots_real = [r.evalf() for r in roots if sp.im(r) == 0]
    plt.scatter(roots_real, [0] * len(roots_real), color="green", marker='o', label="Roots")

    # Critical points
    critical_real = [c.evalf() for c in critical_points if sp.im(c) == 0]
    plt.scatter(critical_real, global_poly(critical_real), color="purple", marker='x', label="Critical Points")
    
    plt.title("Sheaf-Based Polynomial Fitting")
    plt.xlabel("Timestamps")
    plt.ylabel("Values")
    plt.legend()
    plt.show()


# 6. Main workflow
if __name__ == "__main__":
    # Example dataset
    np.random.seed(42)
    timestamps = np.linspace(0, 10, 100)
    values = 3 * timestamps**2 - 4 * timestamps + 5 + np.random.normal(0, 10, size=100)

    # Local polynomial fitting (sheaf perspective)
    degree = 2
    window_size = 20
    local_sections = local_polynomial_fitting_sheaf(timestamps, values, window_size=window_size, degree=degree)

    # Merge local sections into a global polynomial
    global_poly = merge_polynomials(local_sections)

    # Extract Galois-inspired features
    roots, critical_points = extract_features_galois(global_poly)

    # Visualization
    plot_sheaf_results(timestamps, values, global_poly, local_sections, roots, critical_points)
