import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.linalg import lstsq
from scipy.spatial.distance import cdist


# 1. Polynomial fitting for local sheaf sections
def fit_polynomial_sheaf(timestamps, values, degree=3):
    """
    Fits a polynomial of a given degree to a dataset section (local sheaf).
    """
    A = np.vander(timestamps, degree + 1, increasing=False)
    coeffs, _, _, _ = lstsq(A, values)
    poly = np.poly1d(coeffs[::-1])  # Reverse coefficients for np.poly1d
    return poly, coeffs


# 2. Create transition maps for overlapping sections
def compute_transition_maps(local_sections):
    """
    Computes transition maps between overlapping polynomial sections.
    Ensures consistency by minimizing mismatches at overlaps.
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
            diff = poly1(overlap_t) - poly2(overlap_t)
            
            # Compute correction term as a low-order polynomial
            correction_coeffs = np.polyfit(overlap_t, diff, 1)
            correction_poly = np.poly1d(correction_coeffs[::-1])
            transition_maps.append(correction_poly)
        else:
            transition_maps.append(None)
    return transition_maps


# 3. Merge polynomials with derivative matching
def merge_polynomials_with_derivative_matching(local_sections, transition_maps):
    """
    Merges local polynomial sections into a global polynomial with derivative matching.
    """
    merged_coeffs = np.zeros_like(local_sections[0][1])
    weight = np.zeros_like(merged_coeffs)
    
    for i, (poly, coeffs, window_t) in enumerate(local_sections):
        correction = transition_maps[i - 1](window_t) if i > 0 and transition_maps[i - 1] else 0
        corrected_coeffs = coeffs + correction
        merged_coeffs += corrected_coeffs * len(window_t)
        weight += len(window_t)

    return np.poly1d(merged_coeffs / weight)


# 4. Symmetry analysis using automorphisms
def analyze_field_symmetry(poly):
    """
    Detects symmetry properties of the polynomial using Galois-inspired automorphisms.
    """
    x = sp.Symbol('x')
    poly_expr = sp.Poly.from_list(poly.coefficients[::-1], x).as_expr()
    
    # Test for even/odd symmetry
    even_check = sp.simplify(poly_expr - poly_expr.subs(x, -x))
    odd_check = sp.simplify(poly_expr + poly_expr.subs(x, -x))
    
    if even_check == 0:
        symmetry = "Even"
    elif odd_check == 0:
        symmetry = "Odd"
    else:
        symmetry = "No symmetry"

    # Automorphisms: transformation invariants
    automorphisms = []
    for a in range(-3, 4):  # Limited range for exploration
        transformed_expr = poly_expr.subs(x, x + a)
        automorphisms.append((a, sp.simplify(transformed_expr - poly_expr) == 0))

    return symmetry, automorphisms


# 5. Persistent homology analysis of features
def analyze_persistent_homology(roots, critical_points):
    """
    Analyzes topological stability of roots and critical points using persistent homology.
    """
    all_features = roots + critical_points
    persistence_diagram = cdist(np.array(all_features).reshape(-1, 1), np.array(all_features).reshape(-1, 1))
    persistence_values = np.min(persistence_diagram, axis=1)
    return persistence_values


# 6. Visualization improvements
def plot_global_fit_with_symmetry(timestamps, values, global_poly, roots, critical_points, symmetry, automorphisms):
    """
    Visualizes the global polynomial fit, roots, critical points, and symmetry properties.
    """
    plt.figure(figsize=(12, 8))
    
    # Original data
    plt.scatter(timestamps, values, label="Original Data", alpha=0.6)
    
    # Global polynomial fit
    plt.plot(timestamps, global_poly(timestamps), color="red", label="Global Fit", linewidth=2)
    
    # Roots and critical points
    roots_real = [r.evalf() for r in roots if sp.im(r) == 0]
    critical_real = [c.evalf() for c in critical_points if sp.im(c) == 0]
    plt.scatter(roots_real, [0] * len(roots_real), color="green", label="Roots", marker='o')
    plt.scatter(critical_real, global_poly(critical_real), color="purple", label="Critical Points", marker='x')
    
    # Symmetry information
    plt.title(f"Global Fit (Symmetry: {symmetry})")
    plt.xlabel("Timestamps")
    plt.ylabel("Values")
    plt.legend()
    plt.show()
    
    # Automorphisms
    print("Field Automorphisms (Translation Invariance):")
    for a, invariant in automorphisms:
        print(f"  x -> x + {a}: {'Invariant' if invariant else 'Not Invariant'}")


# 7. Main Workflow
if __name__ == "__main__":
    # Example dataset
    np.random.seed(42)
    timestamps = np.linspace(0, 10, 100)
    values = 3 * timestamps**2 - 4 * timestamps + 5 + np.random.normal(0, 10, size=100)

    # Local polynomial fitting
    degree = 2
    window_size = 20
    local_sections = local_polynomial_fitting_sheaf(timestamps, values, window_size=window_size, degree=degree)

    # Compute transition maps
    transition_maps = compute_transition_maps(local_sections)

    # Merge into a global polynomial
    global_poly = merge_polynomials_with_derivative_matching(local_sections, transition_maps)

    # Analyze symmetry
    symmetry, automorphisms = analyze_field_symmetry(global_poly)

    # Extract roots and critical points
    roots, critical_points = extract_features_galois(global_poly)

    # Persistent homology analysis
    persistence = analyze_persistent_homology(roots, critical_points)
    print(f"Persistence Values: {persistence}")

    # Visualization
    plot_global_fit_with_symmetry(timestamps, values, global_poly, roots, critical_points, symmetry, automorphisms)
