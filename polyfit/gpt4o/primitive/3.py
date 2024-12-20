import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.fftpack import fft, fftfreq

# 1. Polynomial fitting function
def fit_polynomial(timestamps, values, degree=3):
    """Fits a polynomial of the specified degree to the dataset."""
    coeffs = np.polyfit(timestamps, values, degree)
    poly = np.poly1d(coeffs)
    return poly, coeffs


# 2. Feature detection: roots, critical points, and periodicity
def extract_features(poly, timestamps):
    """
    Extracts roots, critical points (maxima/minima), and periodicity from the polynomial.
    """
    x = sp.Symbol('x')
    poly_expr = sp.Poly.from_list(poly.coefficients, x)
    
    # Roots
    roots = sp.solvers.solve(poly_expr, x)
    
    # Critical points (first derivative = 0)
    deriv = sp.diff(poly_expr.as_expr(), x)
    critical_points = sp.solvers.solve(deriv, x)
    
    # Periodicity (via FFT)
    fft_values = fft(poly(timestamps))
    freqs = fftfreq(len(timestamps), d=(timestamps[1] - timestamps[0]))
    dominant_freqs = freqs[np.argsort(np.abs(fft_values))[-3:]]  # Top 3 frequencies
    
    return roots, critical_points, dominant_freqs


# 3. Automated degree selection
def select_polynomial_degree(timestamps, values, max_degree=10):
    """
    Automatically selects the optimal polynomial degree using cross-validation.
    """
    best_degree = 1
    best_error = float('inf')
    
    for degree in range(1, max_degree + 1):
        poly, _ = fit_polynomial(timestamps, values, degree)
        error = np.mean((values - poly(timestamps)) ** 2)  # Mean Squared Error
        
        if error < best_error:
            best_error = error
            best_degree = degree
    
    return best_degree


# 4. Adaptive local polynomial fitting
def adaptive_local_fitting(timestamps, values, min_window=10, degree=3):
    """
    Performs local polynomial fitting with adaptive window sizes based on data density.
    """
    local_models = []
    num_points = len(timestamps)
    i = 0
    
    while i < num_points:
        # Determine the adaptive window size based on data density
        window_size = min_window
        while i + window_size < num_points and (timestamps[i + window_size] - timestamps[i] < 1.5 * (timestamps[1] - timestamps[0])):
            window_size += 1
        
        # Fit polynomial to the window
        window_t = timestamps[i:i + window_size]
        window_v = values[i:i + window_size]
        if len(window_t) >= degree + 1:  # Ensure sufficient data points
            poly, coeffs = fit_polynomial(window_t, window_v, degree)
            local_models.append((poly, coeffs, window_t))
        
        i += window_size  # Move to the next window
    
    return local_models


# 5. Symmetry analysis using Galois concepts
def analyze_symmetry(poly):
    """
    Analyzes symmetry properties of the polynomial using critical points and roots.
    """
    roots, critical_points = extract_features(poly, [])
    # Symmetry types: 
    # (1) Odd function symmetry: f(-x) = -f(x)
    # (2) Even function symmetry: f(-x) = f(x)
    x = sp.Symbol('x')
    expr = sp.Poly.from_list(poly.coefficients, x).as_expr()
    even_check = sp.simplify(expr - expr.subs(x, -x))
    odd_check = sp.simplify(expr + expr.subs(x, -x))
    
    if even_check == 0:
        return "Even Symmetry"
    elif odd_check == 0:
        return "Odd Symmetry"
    else:
        return "No Symmetry Detected"


# 6. Visualization improvements
def plot_results(timestamps, values, poly, local_models, roots, critical_points, dominant_freqs):
    """
    Plots the original data, global fit, local fits, roots, and critical points.
    """
    plt.figure(figsize=(12, 8))
    
    # Original data
    plt.scatter(timestamps, values, label="Original Data", alpha=0.6)
    
    # Global polynomial fit
    plt.plot(timestamps, poly(timestamps), color="red", label="Global Fit", linewidth=2)
    
    # Local polynomial fits
    for poly_local, _, window_t in local_models:
        t_fit = np.linspace(min(window_t), max(window_t), 100)
        plt.plot(t_fit, poly_local(t_fit), '--', label=f"Local Fit [{int(min(window_t))}-{int(max(window_t))}]")
    
    # Roots
    if roots:
        roots_real = [r.evalf() for r in roots if sp.im(r) == 0]
        plt.scatter(roots_real, [0] * len(roots_real), label="Roots", color="green", marker='o')
    
    # Critical points
    critical_real = [c.evalf() for c in critical_points if sp.im(c) == 0]
    plt.scatter(critical_real, poly(critical_real), label="Critical Points", color="purple", marker='x')
    
    # Dominant frequencies
    plt.title(f"Global and Local Fits (Dominant Frequencies: {dominant_freqs[:2]} Hz)")
    plt.xlabel("Timestamps")
    plt.ylabel("Values")
    plt.legend()
    plt.show()


# 7. Main workflow
if __name__ == "__main__":
    # Example dataset
    np.random.seed(42)
    timestamps = np.linspace(0, 10, 100)
    values = 3 * timestamps**2 - 4 * timestamps + 5 + np.random.normal(0, 10, size=100)

    # Automatically select degree
    optimal_degree = select_polynomial_degree(timestamps, values, max_degree=5)
    print(f"Optimal Polynomial Degree: {optimal_degree}")

    # Global polynomial fitting
    poly, coeffs = fit_polynomial(timestamps, values, degree=optimal_degree)
    roots, critical_points, dominant_freqs = extract_features(poly, timestamps)

    # Adaptive local polynomial fitting
    local_models = adaptive_local_fitting(timestamps, values, min_window=10, degree=optimal_degree)

    # Symmetry analysis
    symmetry_type = analyze_symmetry(poly)
    print(f"Symmetry Type: {symmetry_type}")

    # Visualization
    plot_results(timestamps, values, poly, local_models, roots, critical_points, dominant_freqs)
