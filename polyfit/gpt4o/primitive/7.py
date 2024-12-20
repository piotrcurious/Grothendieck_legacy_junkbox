import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.linalg import lstsq


# 1. Fit polynomial for local sections
def fit_polynomial_sheaf(timestamps, values, degree=3):
    """
    Fits a polynomial of a given degree to a dataset section (local sheaf).
    """
    A = np.vander(timestamps, degree + 1, increasing=False)
    coeffs, _, _, _ = lstsq(A, values)
    poly = np.poly1d(coeffs[::-1])  # Reverse coefficients for np.poly1d
    return poly, coeffs


# 2. Compute higher-order transition maps for overlapping sections
def compute_higher_order_transition_maps(local_sections, derivative_order=2):
    """
    Computes transition maps between overlapping polynomial sections.
    Matches values and higher-order derivatives for smooth transitions.
    """
    transition_maps = []
    for i in range(len(local_sections) - 1):
        poly1, coeffs1, window1 = local_sections[i]
        poly2, coeffs2, window2 = local_sections[i + 1]
        
        # Overlap region
        overlap_start = max(window1[0], window2[0])
        overlap_end = min(window1[-1], window2[-1])
        
        if overlap_start < overlap_end:
            overlap_t = np.linspace(overlap_start, overlap_end, 50)
            
            # Compute value and derivative mismatches
            value_diff = poly1(overlap_t) - poly2(overlap_t)
            derivative_diffs = [
                np.polyder(poly1, d)(overlap_t) - np.polyder(poly2, d)(overlap_t)
                for d in range(1, derivative_order + 1)
            ]
            
            # Fit correction polynomials for values and derivatives
            correction_coeffs = [np.polyfit(overlap_t, value_diff, 1)]
            correction_coeffs += [np.polyfit(overlap_t, diff, 1) for diff in derivative_diffs]
            
            correction_polys = [np.poly1d(coeffs[::-1]) for coeffs in correction_coeffs]
            transition_maps.append(correction_polys)
        else:
            transition_maps.append(None)
    return transition_maps


# 3. Merge sections using higher-order derivative constraints
def merge_polynomials_with_higher_order_matching(local_sections, transition_maps):
    """
    Merges local polynomial sections into a global polynomial with higher-order matching.
    """
    merged_coeffs = np.zeros_like(local_sections[0][1])
    weight = np.zeros_like(merged_coeffs)
    
    for i, (poly, coeffs, window_t) in enumerate(local_sections):
        if i > 0 and transition_maps[i - 1]:
            corrections = transition_maps[i - 1]
            correction_terms = sum(c(np.mean(window_t)) for c in corrections)
            corrected_coeffs = coeffs + correction_terms
        else:
            corrected_coeffs = coeffs

        merged_coeffs += corrected_coeffs * len(window_t)
        weight += len(window_t)

    return np.poly1d(merged_coeffs / weight)


# 4. Extract roots and critical points
def extract_features(poly):
    """
    Extracts roots and critical points of the polynomial.
    """
    x = sp.Symbol('x')
    poly_expr = sp.Poly.from_list(poly.coefficients[::-1], x)
    
    # Roots
    roots = sp.solvers.solve(poly_expr, x)

    # Critical points (first derivative = 0)
    deriv = sp.diff(poly_expr.as_expr(), x)
    critical_points = sp.solvers.solve(deriv, x)

    return roots, critical_points


# 5. Local polynomial fitting with adaptive degree
def local_polynomial_fitting_adaptive(timestamps, values, window_size=10, base_degree=3):
    """
    Fits local polynomial sections to the dataset using sliding windows.
    Adjusts the degree of the polynomial based on local variance in the data.
    """
    local_sections = []
    num_points = len(timestamps)

    for i in range(0, num_points, window_size):
        # Extract the window
        window_t = timestamps[i:i + window_size]
        window_v = values[i:i + window_size]

        if len(window_t) >= base_degree + 1:  # Ensure sufficient data points
            local_variance = np.var(window_v)
            degree = base_degree + (1 if local_variance > 1 else 0)  # Adjust degree
            poly, coeffs = fit_polynomial_sheaf(window_t, window_v, degree)
            local_sections.append((poly, coeffs, window_t))

    return local_sections


# 6. Visualization with transition maps
def plot_results(timestamps, values, global_poly, local_sections, roots, critical_points):
    """
    Visualizes the dataset, global polynomial, local fits, roots, and critical points.
    """
    plt.figure(figsize=(14, 8))
    
    # Original data
    plt.scatter(timestamps, values, label="Original Data", alpha=0.6)
    
    # Global polynomial fit
    plt.plot(timestamps, global_poly(timestamps), color="red", label="Global Fit", linewidth=2)
    
    # Local fits
    for poly, _, window_t in local_sections:
        t_fit = np.linspace(min(window_t), max(window_t), 100)
        plt.plot(t_fit, poly(t_fit), '--', label=f"Local Fit [{int(min(window_t))}-{int(max(window_t))}]")
    
    # Roots
    roots_real = [r.evalf() for r in roots if sp.im(r) == 0]
    plt.scatter(roots_real, [0] * len(roots_real), color="green", label="Roots", marker='o')
    
    # Critical points
    critical_real = [c.evalf() for c in critical_points if sp.im(c) == 0]
    plt.scatter(critical_real, global_poly(critical_real), color="purple", label="Critical Points", marker='x')
    
    plt.title("Global and Local Polynomial Fits with Higher-Order Matching")
    plt.xlabel("Timestamps")
    plt.ylabel("Values")
    plt.legend()
    plt.show()


# 7. Main Workflow
if __name__ == "__main__":
    # Example dataset
    np.random.seed(42)
    timestamps = np.linspace(0, 10, 100)
    values = 3 * timestamps**2 - 4 * timestamps + 5 + np.random.normal(0, 10, size=100)

    # Adaptive local polynomial fitting
    base_degree = 2
    window_size = 20
    local_sections = local_polynomial_fitting_adaptive(timestamps, values, window_size=window_size, base_degree=base_degree)

    # Compute transition maps
    transition_maps = compute_higher_order_transition_maps(local_sections, derivative_order=2)

    # Merge into a global polynomial
    global_poly = merge_polynomials_with_higher_order_matching(local_sections, transition_maps)

    # Extract features
    roots, critical_points = extract_features(global_poly)

    # Visualization
    plot_results(timestamps, values, global_poly, local_sections, roots, critical_points)
