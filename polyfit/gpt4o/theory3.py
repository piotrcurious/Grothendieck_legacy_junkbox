import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, Poly, solve, simplify, expand
from sympy.abc import x

# 1. Dataset: Representing the data as elements of a field
class DataPoint:
    """Represents a data point in a field (timestamp, value)."""
    def __init__(self, timestamp, value):
        self.timestamp = timestamp
        self.value = value
    
    def __repr__(self):
        return f"({self.timestamp}, {self.value})"

# Example dataset
data = [DataPoint(0, 2), DataPoint(1, 3), DataPoint(2, 5), 
        DataPoint(3, 7), DataPoint(4, 11), DataPoint(5, 13)]

# Extract timestamps and values
timestamps = np.array([point.timestamp for point in data])
values = np.array([point.value for point in data])

# 2. Polynomial fitting as a morphism
degree = 3
coefficients = np.polyfit(timestamps, values, degree)

# Create a polynomial as a categorical morphism
def polynomial_morphism(x):
    return sum(c * x**i for i, c in enumerate(reversed(coefficients)))

polynomial = Poly(polynomial_morphism(x), x)

# 3. Feature extraction (roots and symmetries)
# Roots: Where polynomial evaluates to zero
roots = solve(polynomial, x)

# Symmetry (invariants under transformations): Polynomial coefficients
symmetry_invariants = polynomial.coeffs()

# Functorial mapping: Original data to polynomial space
mapped_data = [polynomial_morphism(pt.timestamp) for pt in data]

# 4. Visualization: Abstract morphism to geometric representation
x_vals = np.linspace(min(timestamps) - 1, max(timestamps) + 1, 500)
y_vals = [polynomial_morphism(x_val) for x_val in x_vals]

plt.figure(figsize=(10, 6))
plt.scatter(timestamps, values, color='red', label='Original Data (Field Elements)')
plt.plot(x_vals, y_vals, label=f'Polynomial Morphism (Degree {degree})', color='blue')
plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
plt.axvline(0, color='black', linestyle='--', linewidth=0.5)

# Annotating roots as fixed points in the morphism
for root in roots:
    plt.scatter(float(root), 0, color='green', label=f'Root: {root.evalf():.2f}')

plt.title('Polynomial Fitting as Morphism and Feature Extraction')
plt.xlabel('Timestamps')
plt.ylabel('Values')
plt.legend()
plt.grid()
plt.show()

# 5. Print Results: Abstract algebra meets computation
print("Original Data (Field Elements):", data)
print("Polynomial Coefficients (Symmetry Invariants):", symmetry_invariants)
print("Polynomial (Categorical Morphism):", polynomial)
print("Simplified Polynomial:", simplify(polynomial))
print("Roots (Fixed Points):", roots)
print("Mapped Data in Polynomial Space:", mapped_data)
