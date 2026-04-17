import numpy as np import matplotlib.pyplot as plt from matplotlib.widgets import Slider from abc import ABC, abstractmethod from typing import Dict import sympy as sp

##############################################

Core Mathematical Structures and Operators

##############################################

class ModularPhaseSpace: """ Represents a discrete phase space isomorphic to a cyclic group ℤ_modulus. """ def init(self, dimension: int, modulus: int) -> None: if modulus <= 0: raise ValueError("Modulus must be a positive integer (defining the cyclic group Z_modulus).") self.dimension = dimension self.modulus = modulus self.indices = np.arange(dimension) % modulus

def shift_index(self, index: int, shift: int) -> int:
    return (index + shift) % self.modulus

def __repr__(self) -> str:
    return f"ModularPhaseSpace(dimension={self.dimension}, modulus={self.modulus})"

class Operator(ABC): def init(self, phase_space: ModularPhaseSpace) -> None: if not isinstance(phase_space, ModularPhaseSpace): raise TypeError("Operator must be initialized with a ModularPhaseSpace object.") self.phase_space = phase_space

@abstractmethod
def operate(self, matrix: np.ndarray) -> np.ndarray:
    pass

def __call__(self, matrix: np.ndarray) -> np.ndarray:
    return self.operate(matrix)

class ModularWeylXOperator(Operator): def init(self, phase_space: ModularPhaseSpace, quantization_level: int = 1) -> None: super().init(phase_space) self.quantization_level = quantization_level

def operate(self, matrix: np.ndarray) -> np.ndarray:
    if matrix.shape != (self.phase_space.dimension, self.phase_space.dimension):
        raise ValueError("Input must be a square NumPy matrix of specified dimension.")
    shifted = np.zeros_like(matrix)
    for i in range(self.phase_space.dimension):
        si = self.phase_space.shift_index(i, self.quantization_level)
        shifted[si, :] = matrix[i, :]
    return shifted

class ModularWeylPOperator(Operator): def init(self, phase_space: ModularPhaseSpace, quantization_level: int = 1) -> None: super().init(phase_space) self.quantization_level = quantization_level

def operate(self, matrix: np.ndarray) -> np.ndarray:
    if matrix.shape != (self.phase_space.dimension, self.phase_space.dimension):
        raise ValueError("Input must be a square NumPy matrix of specified dimension.")
    shifted = np.zeros_like(matrix)
    for j in range(self.phase_space.dimension):
        sj = self.phase_space.shift_index(j, self.quantization_level)
        shifted[:, sj] = matrix[:, j]
    return shifted

##############################################

Discrete Rotation & Quality Estimation

##############################################

def discrete_rotation_operator(angle_degrees: float, dim: int) -> np.ndarray: theta = np.radians(angle_degrees) R2 = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]) R = np.eye(dim) if dim >= 2: R[:2, :2] = R2 return R

def apply_rotation(M: np.ndarray, R: np.ndarray) -> np.ndarray: return R @ M @ R.T

def estimate_quantization_effect_modular(M: np.ndarray, X: Operator, P: Operator) -> float: xp = P(X(M.copy())) px = X(P(M.copy())) return np.linalg.norm(xp - px, 'fro')

##############################################

Correction Strategies

##############################################

class CorrectionStrategy(ABC): @abstractmethod def get_correction_matrix(self, initial: np.ndarray, angle: float, weyl_x: Operator, weyl_p: Operator): pass

class SymbolicPolynomialCorrectionStrategy(CorrectionStrategy): """ Builds correction matrices as symbolic polynomials in a ring R[x]/(p(x)). """ def init(self, modulus: int, poly_coeffs: list, alpha: float = 0.01): self.modulus = modulus self.alpha = alpha self.x = sp.symbols('x') # minimal polynomial p(x) self.p = sp.Poly(poly_coeffs, self.x, domain=sp.GF(modulus))

def get_correction_matrix(self, initial: np.ndarray, angle: float,
                          weyl_x: Operator, weyl_p: Operator):
    dim = initial.shape[0]
    q = estimate_quantization_effect_modular(initial, weyl_x, weyl_p)
    # build basis: 1, x, x^2, ..., x^{dim-1} mod p(x)
    basis = []
    for i in range(dim):
        poly = (self.x**i) % self.p
        basis.append(sp.Poly(poly, self.x, domain=sp.GF(self.modulus)))
    # construct multiplication-by-x matrix in the quotient ring
    M = sp.zeros(dim)
    for col, b in enumerate(basis):
        prod = (b * sp.Poly(self.x, self.x, domain=sp.GF(self.modulus))).rem(self.p)
        coeffs = prod.all_coeffs()[::-1]
        for row, c in enumerate(coeffs):
            if row < dim:
                M[row, col] = c
    # incorporate correction factor
    corr = sp.eye(dim) + self.alpha * q * M
    return corr

##############################################

Demonstrating Runge's Phenomenon

##############################################

def runge_function(x): return 1 / (1 + x**2)

def interpolate_runge(n): # equidistant nodes on [-1,1] xs = np.linspace(-1, 1, n+1) ys = runge_function(xs) # find coefficients of interpolating polynomial in monomial basis coeffs = np.polyfit(xs, ys, n) poly = np.poly1d(coeffs) # sample dense grid x_dense = np.linspace(-1,1,500) y_true = runge_function(x_dense) y_interp = poly(x_dense) err = np.max(np.abs(y_true - y_interp)) return x_dense, y_true, y_interp, err

if name == 'main': # Example usage of symbolic correction dim, mod = 5, 7 phase_space = ModularPhaseSpace(dim, mod) wx = ModularWeylXOperator(phase_space) wp = ModularWeylPOperator(phase_space) shape = np.eye(dim) R = discrete_rotation_operator(30, dim) strat = SymbolicPolynomialCorrectionStrategy(mod, [1,0,1,1], 0.02) sym_corr = strat.get_correction_matrix(shape, 30, wx, wp) print("Symbolic Correction Matrix:") sp.pprint(sym_corr)

# Runge phenomenon for degrees 4, 8, 12, 16
for n in [4, 8, 12, 16]:
    xg, y_true, y_int, err = interpolate_runge(n)
    plt.figure(); plt.plot(xg, y_true, label='True')
    plt.plot(xg, y_int, label=f'Interp n={n}')
    plt.title(f'Runge Interpolation Error n={n}, max err={err:.3e}')
    plt.legend()
plt.show()

