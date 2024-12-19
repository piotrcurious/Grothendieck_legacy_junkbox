import numpy as np
import pandas as pd
from scipy.special import legendre
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from scipy.linalg import companion
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns
from typing import List, Tuple, Dict

class GaloisFeatureExtractor:
    """
    Feature extractor based on Galois and Grothendieck theory principles:
    - Treats data as elements of field extensions
    - Uses polynomial bases for feature extraction
    - Implements field-theoretic operations for feature transformation
    """
    def __init__(self, degree: int = 3, n_features: int = 5):
        self.degree = degree
        self.n_features = n_features
        self.poly_features = PolynomialFeatures(degree=degree)
        self.pca = PCA(n_components=n_features)
        self.legendre_coeffs = {}
        
    def compute_minimal_polynomial(self, values: np.ndarray) -> np.ndarray:
        """
        Compute approximate minimal polynomial using companion matrix.
        This represents the field extension basis.
        """
        n = len(values)
        if n < 2:
            return values
        
        # Create Hankel matrix
        hankel = np.zeros((n//2, n//2))
        for i in range(n//2):
            for j in range(n//2):
                if i + j < n:
                    hankel[i,j] = values[i+j]
                    
        # Compute companion matrix coefficients
        _, _, Vh = np.linalg.svd(hankel)
        coeffs = Vh[-1]
        return coeffs
    
    def generate_field_basis(self, x: np.ndarray) -> np.ndarray:
        """
        Generate basis for field extension using Legendre polynomials.
        This implements Grothendieck's relative point of view.
        """
        basis = []
        for i in range(self.degree + 1):
            leg_poly = legendre(i)
            self.legendre_coeffs[i] = leg_poly.coeffs
            basis.append(leg_poly(x))
        return np.array(basis).T
    
    def galois_orbit_features(self, x: np.ndarray, values: np.ndarray) -> np.ndarray:
        """
        Extract features based on Galois orbit principle.
        Implements the idea that features should respect field automorphisms.
        """
        # Generate polynomial basis
        basis = self.generate_field_basis(x)
        
        # Transform through polynomial features (representing field extension)
        poly_features = self.poly_features.fit_transform(basis)
        
        # Apply PCA to find principal Galois-invariant features
        return self.pca.fit_transform(poly_features)
    
    def scheme_morphism(self, features: np.ndarray) -> np.ndarray:
        """
        Implement scheme morphism for dimension reduction.
        This represents Grothendieck's geometric perspective.
        """
        # Compute field trace analog
        trace = np.mean(features, axis=1)
        
        # Compute field norm analog
        norm = np.prod(np.abs(features) + 1, axis=1)
        
        return np.vstack([trace, norm]).T

class PolynomialFieldFitter:
    """
    Polynomial fitting using field extension principles.
    """
    def __init__(self, degree: int = 3):
        self.degree = degree
        self.coefficients = None
        
    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        Fit polynomial using field extension basis.
        """
        basis = np.vstack([x**i for i in range(self.degree + 1)]).T
        self.coefficients = np.linalg.lstsq(basis, y, rcond=None)[0]
        
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict using fitted polynomial.
        """
        basis = np.vstack([x**i for i in range(self.degree + 1)]).T
        return basis @ self.coefficients

def generate_example_data(n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate example time series data with multiple frequencies.
    """
    timestamps = np.linspace(0, 10, n_points)
    values = (np.sin(timestamps) + 
              0.5 * np.sin(3 * timestamps) + 
              0.2 * np.random.randn(n_points))
    return timestamps, values

def visualize_results(timestamps: np.ndarray, 
                     values: np.ndarray,
                     features: np.ndarray,
                     fitted_values: np.ndarray):
    """
    Visualize the original data, extracted features, and fitted polynomial.
    """
    fig = plt.figure(figsize=(15, 10))
    
    # Original data and polynomial fit
    ax1 = plt.subplot(211)
    ax1.plot(timestamps, values, 'b.', label='Original Data')
    ax1.plot(timestamps, fitted_values, 'r-', label='Polynomial Fit')
    ax1.set_title('Time Series Data and Polynomial Fit')
    ax1.legend()
    
    # Feature visualization
    ax2 = plt.subplot(212)
    for i in range(features.shape[1]):
        ax2.plot(timestamps, features[:, i], 
                label=f'Feature {i+1}')
    ax2.set_title('Extracted Galois-Invariant Features')
    ax2.legend()
    
    plt.tight_layout()
    return fig

def main():
    # Generate example data
    timestamps, values = generate_example_data(100)
    
    # Normalize time to [-1, 1] for numerical stability
    timestamps_norm = 2 * (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min()) - 1
    
    # Extract features
    feature_extractor = GaloisFeatureExtractor(degree=3, n_features=3)
    features = feature_extractor.galois_orbit_features(timestamps_norm, values)
    
    # Fit polynomial
    fitter = PolynomialFieldFitter(degree=5)
    fitter.fit(timestamps_norm, values)
    fitted_values = fitter.predict(timestamps_norm)
    
    # Visualize results
    fig = visualize_results(timestamps, values, features, fitted_values)
    plt.show()
    
    # Print some analytical results
    minimal_poly = feature_extractor.compute_minimal_polynomial(values[:10])
    print("\nApproximate minimal polynomial coefficients:")
    print(minimal_poly)
    
    print("\nLegendre polynomial coefficients used in basis:")
    for deg, coeffs in feature_extractor.legendre_coeffs.items():
        print(f"Degree {deg}: {coeffs}")

if __name__ == "__main__":
    main()
