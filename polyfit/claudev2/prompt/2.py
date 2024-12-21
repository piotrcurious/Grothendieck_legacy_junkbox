import numpy as np
from typing import List, Tuple, Union
from dataclasses import dataclass
from functools import reduce
from operator import mul

@dataclass
class F2Polynomial:
    """
    Represents a polynomial over F2 (binary field)
    coefficients: List of coefficients in F2 (0 or 1)
    """
    coefficients: List[int]
    
    def __post_init__(self):
        self.coefficients = [c % 2 for c in self.coefficients]
        # Remove trailing zeros
        while self.coefficients and self.coefficients[-1] == 0:
            self.coefficients.pop()
    
    def __add__(self, other):
        # Addition in F2 is XOR
        max_len = max(len(self.coefficients), len(other.coefficients))
        result = [0] * max_len
        for i in range(max_len):
            a = self.coefficients[i] if i < len(self.coefficients) else 0
            b = other.coefficients[i] if i < len(other.coefficients) else 0
            result[i] = (a + b) % 2
        return F2Polynomial(result)
    
    def __mul__(self, other):
        # Multiplication in F2[x]
        result = [0] * (len(self.coefficients) + len(other.coefficients))
        for i, a in enumerate(self.coefficients):
            for j, b in enumerate(other.coefficients):
                result[i + j] ^= a & b  # XOR for addition in F2
        return F2Polynomial(result)

class SchemeStructure:
    """
    Represents a scheme structure for numeric types
    """
    def __init__(self, bit_width: int, is_signed: bool = True, is_float: bool = False):
        self.bit_width = bit_width
        self.is_signed = is_signed
        self.is_float = is_float
        
        if is_float:
            # IEEE-754 structure
            self.mantissa_bits = {32: 23, 64: 52}[bit_width]
            self.exponent_bits = {32: 8, 64: 11}[bit_width]
        
    def to_polynomial(self, value: Union[int, float]) -> F2Polynomial:
        """Convert a numeric value to its F2 polynomial representation"""
        if self.is_float:
            return self._float_to_polynomial(value)
        else:
            return self._int_to_polynomial(value)
    
    def _int_to_polynomial(self, value: int) -> F2Polynomial:
        """Convert integer to F2 polynomial"""
        if self.is_signed:
            value = value % (2 ** self.bit_width)
        binary = bin(abs(value))[2:].zfill(self.bit_width)
        return F2Polynomial([int(b) for b in binary[::-1]])
    
    def _float_to_polynomial(self, value: float) -> F2Polynomial:
        """Convert float to F2 polynomial using IEEE-754 structure"""
        bits = np.frombuffer(np.array([value], dtype=f'float{self.bit_width}').tobytes(), dtype='uint8')
        binary = ''.join(format(b, '08b') for b in bits)
        return F2Polynomial([int(b) for b in binary[::-1]])

class PolynomialFeatureExtractor:
    """
    Extracts polynomial features while preserving field structure
    """
    def __init__(self, degree: int, scheme: SchemeStructure):
        self.degree = degree
        self.scheme = scheme
        
    def _generate_monomials(self, poly: F2Polynomial, degree: int) -> List[F2Polynomial]:
        """Generate monomials up to given degree"""
        result = [F2Polynomial([1])]  # Constant term
        current = poly
        for _ in range(degree):
            result.append(current)
            current = current * poly
        return result
    
    def extract_features(self, time_series: List[Tuple[int, Union[int, float]]]) -> np.ndarray:
        """
        Extract polynomial features from time series data
        time_series: List of (timestamp, value) pairs
        """
        features = []
        for timestamp, value in time_series:
            # Convert value to F2 polynomial
            value_poly = self.scheme.to_polynomial(value)
            
            # Generate time polynomial (using Frobenius endomorphism structure)
            time_poly = self.scheme.to_polynomial(timestamp)
            time_features = self._generate_monomials(time_poly, self.degree)
            
            # Generate value features
            value_features = self._generate_monomials(value_poly, self.degree)
            
            # Combine features using tensor product structure
            combined = []
            for t_feat in time_features:
                for v_feat in value_features:
                    combined.extend(t_feat.coefficients)
                    combined.extend(v_feat.coefficients)
            
            features.append(combined)
        
        return np.array(features)

class SchemePreservingRegressor:
    """
    Implements regression while respecting scheme morphisms
    """
    def __init__(self, scheme: SchemeStructure, learning_rate: float = 0.01):
        self.scheme = scheme
        self.learning_rate = learning_rate
        self.weights = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100):
        """
        Fit the model while preserving scheme structure
        """
        n_features = X.shape[1]
        self.weights = np.random.randn(n_features) / np.sqrt(n_features)
        
        for _ in range(epochs):
            # Forward pass
            y_pred = X @ self.weights
            
            # Convert predictions to scheme structure
            y_pred_poly = [self.scheme.to_polynomial(val) for val in y_pred]
            y_true_poly = [self.scheme.to_polynomial(val) for val in y]
            
            # Compute gradients respecting field structure
            grad = np.zeros_like(self.weights)
            for i in range(len(X)):
                error_poly = y_pred_poly[i] + y_true_poly[i]  # Addition in F2
                grad += error_poly.coefficients[0] * X[i]
            
            # Update weights
            self.weights -= self.learning_rate * grad / len(X)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions respecting scheme structure"""
        return X @ self.weights

# Example usage
def example_usage():
    # Create scheme structure for float64
    scheme = SchemeStructure(bit_width=64, is_float=True)
    
    # Create feature extractor
    extractor = PolynomialFeatureExtractor(degree=2, scheme=scheme)
    
    # Generate sample time series data
    time_series = [(i, np.sin(i/10)) for i in range(100)]
    
    # Extract features
    features = extractor.extract_features(time_series)
    
    # Create and train regressor
    regressor = SchemePreservingRegressor(scheme)
    X = features[:-1]
    y = np.array([v for _, v in time_series[1:]])
    regressor.fit(X, y)
    
    # Make predictions
    predictions = regressor.predict(X)
    return predictions
