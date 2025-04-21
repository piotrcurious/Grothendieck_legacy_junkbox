import numpy as np import matplotlib.pyplot as plt from abc import ABC, abstractmethod from typing import Dict, Any, List, Optional, Union

##############################################

Core Mathematical Structures and Operators

##############################################

class ModularPhaseSpace: r""" Represents a discrete phase space isomorphic to Z_modulus. """ def init(self, dimension: int, modulus: int) -> None: if modulus <= 0: raise ValueError("Modulus must be a positive integer.") self.dimension = dimension self.modulus = modulus self.indices = np.arange(dimension) % modulus

def get_index(self, index: int) -> int:
    return index % self.modulus

def shift_index(self, index: int, shift: int) -> int:
    return (index + shift) % self.modulus

def __repr__(self) -> str:
    return f"ModularPhaseSpace(dimension={self.dimension}, modulus={self.modulus})"

class Operator(ABC): def init(self, phase_space: ModularPhaseSpace) -> None: if not isinstance(phase_space, ModularPhaseSpace): raise TypeError("Operator must be initialized with a ModularPhaseSpace.") self.phase_space = phase_space

@abstractmethod
def operate(self, matrix: np.ndarray) -> np.ndarray:
    pass

def __call__(self, matrix: np.ndarray) -> np.ndarray:
    return self.operate(matrix)

class ModularWeylXOperator(Operator): def init(self, phase_space: ModularPhaseSpace, quantization_level: int = 1) -> None: super().init(phase_space) self.quantization_level = quantization_level

def operate(self, matrix: np.ndarray) -> np.ndarray:
    if matrix.shape != (self.phase_space.dimension, self.phase_space.dimension):
        raise ValueError("Matrix must be square of correct dimension.")
    out = np.zeros_like(matrix)
    for i in range(self.phase_space.dimension):
        out[self.phase_space.shift_index(i, self.quantization_level), :] = matrix[i, :]
    return out

class ModularWeylPOperator(Operator): def init(self, phase_space: ModularPhaseSpace, quantization_level: int = 1) -> None: super().init(phase_space) self.quantization_level = quantization_level

def operate(self, matrix: np.ndarray) -> np.ndarray:
    if matrix.shape != (self.phase_space.dimension, self.phase_space.dimension):
        raise ValueError("Matrix must be square of correct dimension.")
    out = np.zeros_like(matrix)
    for j in range(self.phase_space.dimension):
        out[:, self.phase_space.shift_index(j, self.quantization_level)] = matrix[:, j]
    return out

def discrete_rotation_operator(angle_degrees: float, dim: int) -> np.ndarray: theta = np.radians(angle_degrees) R2 = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta),  np.cos(theta)]], dtype=float) R = np.eye(dim, dtype=float) if dim >= 2: R[:2, :2] = R2 return R

def apply_discrete_rotation(matrix: np.ndarray, R: np.ndarray) -> np.ndarray: if matrix.shape != R.shape: raise ValueError("Shapes must match.") return R @ matrix @ R.T

def estimate_quantization_effect_modular(matrix: np.ndarray, weyl_x: Operator, weyl_p: Operator) -> float: xp = weyl_p(weyl_x(matrix.copy())) px = weyl_x(weyl_p(matrix.copy())) return np.linalg.norm(xp - px, 'fro')

##############################################

Correction Strategies

##############################################

class CorrectionStrategy(ABC): @abstractmethod def get_correction_matrix(self, matrix: np.ndarray, angle_deg: float, weyl_x: Operator, weyl_p: Operator) -> np.ndarray: pass

class LemmaBasedCorrectionStrategy(CorrectionStrategy): def init(self, correction_factor: float = 0.005) -> None: self.correction_factor = correction_factor

def get_correction_matrix(self, matrix: np.ndarray, angle_deg: float,
                          weyl_x: Operator, weyl_p: Operator) -> np.ndarray:
    Q = estimate_quantization_effect_modular(matrix, weyl_x, weyl_p)
    scale = 1.0 - self.correction_factor * Q
    return np.eye(matrix.shape[0]) * scale

class RandomNoiseCorrectionStrategy(CorrectionStrategy): def init(self, noise_factor: float = 0.01) -> None: self.noise_factor = noise_factor

def get_correction_matrix(self, matrix: np.ndarray, angle_deg: float,
                          weyl_x: Operator, weyl_p: Operator) -> np.ndarray:
    noise = (np.random.rand(*matrix.shape) - 0.5) * self.noise_factor
    return np.eye(matrix.shape[0]) + noise

def apply_modular_corrected_rotation(matrix: np.ndarray, R: np.ndarray, C: np.ndarray) -> np.ndarray: if matrix.shape != R.shape or matrix.shape != C.shape: raise ValueError("Shapes must match.") return R @ (C @ matrix) @ R.T

##############################################

Helper Functions

##############################################

def create_shape_matrix(shape_type: str, dim: int) -> np.ndarray: mat = np.zeros((dim, dim), dtype=float) cx, cy = dim // 2, dim // 2 if shape_type == 'square': size = dim // 3 s = cx - size // 2 mat[s:s+size, s:s+size] = 1.0 elif shape_type == 'circle': X, Y = np.ogrid[:dim, :dim] mask = (X - cx)**2 + (Y - cy)**2 <= (dim//4)**2 mat[mask] = 1.0 elif shape_type == 'triangle': for i in range(cx, dim): width = (dim - i) * dim / (2 * dim) j0 = int(cx - width) j1 = int(cx + width) mat[i, j0:j1] = 1.0 elif shape_type == 'line': w = max(1, dim // 8) mat[cx-w//2:cx+w//2, dim//4:3*dim//4] = 1.0 elif shape_type == 'cross': w = max(1, dim // 5) mat[cx-w//2:cx+w//2, :] = 1.0 mat[:, cy-w//2:cy+w//2] = 1.0 else: raise ValueError(f"Unsupported shape_type: {shape_type}") return mat

def calculate_rotation_quality(original: np.ndarray, standard: np.ndarray, corrected: np.ndarray) -> Dict[str, float]: overlap_std = float(np.sum(original * standard)) overlap_corr = float(np.sum(original * corrected)) diff_norm = float(np.linalg.norm(corrected - standard, 'fro')) return { 'overlap_standard': overlap_std, 'overlap_corrected': overlap_corr, 'difference_norm': diff_norm, }

##############################################

Experiment and Visualization

##############################################

def display_matrices_graphical(std: np.ndarray, corr: np.ndarray, angle: float, quant: int, c_factor: float, modulus: int, metrics: Dict[str, float], strategy: str) -> Any: fig, axes = plt.subplots(1, 2, figsize=(14, 7)) axes[0].imshow(std, cmap='gray', interpolation='nearest') axes[0].set_title(f"Standard Rotation\nOverlap: {metrics['overlap_standard']:.2f}") axes[0].axis('off')

axes[1].imshow(corr, cmap='gray', interpolation='nearest')
axes[1].set_title(
    f"Corrected ({strategy})\nOverlap: {metrics['overlap_corrected']:.2f}\n"
    f"Diff Norm: {metrics['difference_norm']:.2f}")
axes[1].axis('off')

fig.suptitle(
    f"Angle: {angle}°, Quant: {quant}, Corr Fact: {c_factor:.4f}, Modulus: {modulus}"
)
plt.tight_layout(rect=[0,0.03,1,0.95])
return fig

def run_experiment_parameter_variation(dim: int, modulus: int, fixed_quantization_level: int, fixed_rotation_angle_deg: float, correction_strategy: CorrectionStrategy, param_name: str, param_values: List[Union[int, float]], shape_type: str) -> List[Dict[str, float]]: results = [] for val in param_values: q = val if param_name == 'quantization_level' else fixed_quantization_level angle = val if param_name == 'rotation_angle_deg' else fixed_rotation_angle_deg pspace = ModularPhaseSpace(dim, modulus) wx = ModularWeylXOperator(pspace, q) wp = ModularWeylPOperator(pspace, q) R = discrete_rotation_operator(angle, dim) M = create_shape_matrix(shape_type, dim) std = apply_discrete_rotation(M, R) C = correction_strategy.get_correction_matrix(M, angle, wx, wp) corr = apply_modular_corrected_rotation(M, R, C) metrics = calculate_rotation_quality(M, std, corr) metrics[param_name] = float(val) results.append(metrics) return results

def plot_experiment_results(results: List[Dict[str, float]], param_name: str, modulus: int, strategy_name: str) -> None: xs = [r[param_name] for r in results] ys_std = [r['overlap_standard'] for r in results] ys_corr = [r['overlap_corrected'] for r in results] ys_diff = [r['difference_norm'] for r in results]

plt.figure()
plt.plot(xs, ys_std, label='Standard Overlap')
plt.plot(xs, ys_corr, label='Corrected Overlap')
plt.plot(xs, ys_diff, label='Difference Norm')
plt.xlabel(param_name)
plt.ylabel('Metric')
plt.title(f'Experiment vs {param_name} (Modulus={modulus}, Strategy={strategy_name})')
plt.legend()
plt.show()

if name == "main": # Example usage dim = 50 modulus = 50 quant = 2 angle = 30.0 corr_factor = 0.005 shape = 'circle'

# Demonstration
strategy = LemmaBasedCorrectionStrategy(correction_factor=corr_factor)
pspace = ModularPhaseSpace(dim, modulus)
wx = ModularWeylXOperator(pspace, quant)
wp = ModularWeylPOperator(pspace, quant)
R = discrete_rotation_operator(angle, dim)
M = create_shape_matrix(shape, dim)
std = apply_discrete_rotation(M, R)
C = strategy.get_correction_matrix(M, angle, wx, wp)
corr = apply_modular_corrected_rotation(M, R, C)
metrics = calculate_rotation_quality(M, std, corr)
fig = display_matrices_graphical(std, corr, angle, quant, corr_factor, modulus, metrics, strategy.__class__.__name__)
plt.show()

