import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Tuple
import scipy.ndimage
from numpy.typing import NDArray

##############################################
# Core Mathematical Structures and Operators #
##############################################

class ModularPhaseSpace:
    """
    Represents a discrete phase space that is isomorphic to a cyclic group Z_modulus.
    """
    def __init__(self, dimension: int, modulus: int) -> None:
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("Dimension must be a positive integer.")
        if not isinstance(modulus, int) or modulus <= 0:
            raise ValueError("Modulus must be a positive integer.")
        self.dimension: int = dimension
        self.modulus: int = modulus

    def get_index(self, index: int) -> int:
        return index % self.modulus

    def shift_index(self, index: int, shift: int) -> int:
        return (index + shift) % self.modulus

    def __repr__(self) -> str:
        return f"ModularPhaseSpace(dimension={self.dimension}, modulus={self.modulus})"


class Operator(ABC):
    """
    Abstract base class representing an operator on a modular phase space.
    """
    def __init__(self, phase_space: ModularPhaseSpace) -> None:
        if not isinstance(phase_space, ModularPhaseSpace):
            raise TypeError("Operator must be initialized with a ModularPhaseSpace object.")
        self.phase_space: ModularPhaseSpace = phase_space

    @abstractmethod
    def operate(self, matrix: NDArray) -> NDArray:
        pass

    def __call__(self, matrix: NDArray) -> NDArray:
        return self.operate(matrix)

    def __repr__(self) -> str:
        params = [f"{k}={v}" for k, v in self.__dict__.items()
                  if hasattr(self, k) and k != 'phase_space' and not k.startswith('_')]
        return f"{self.__class__.__name__}({', '.join(params)})"


class ModularWeylXOperator(Operator):
    """
    Implements a discrete translation (shift) operator in the X direction (rows).
    X |j> = |j + q mod N>
    """
    def __init__(self, phase_space: ModularPhaseSpace, quantization_level: int = 1) -> None:
        super().__init__(phase_space)
        self.quantization_level: int = quantization_level % self.phase_space.modulus

    def operate(self, matrix: NDArray) -> NDArray:
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Matrix must be square.")
        return np.roll(matrix, shift=self.quantization_level, axis=0)


class ModularWeylPOperator(Operator):
    """
    Implements a discrete clock (phase) operator.
    P |j> = exp(2*pi*i * j * q / N) |j>
    """
    def __init__(self, phase_space: ModularPhaseSpace, quantization_level: int = 1) -> None:
        super().__init__(phase_space)
        self.quantization_level: int = quantization_level % self.phase_space.modulus
        N = self.phase_space.dimension
        j = np.arange(N)
        self.phases = np.exp(2j * np.pi * j * self.quantization_level / N).reshape(-1, 1)

    def operate(self, matrix: NDArray) -> NDArray:
        return matrix * self.phases


def estimate_quantization_effect_modular(matrix: NDArray, weyl_x: Operator, weyl_p: Operator) -> float:
    """Estimates quantization effect norm: Q = || [P, X] ||_F"""
    px_matrix = weyl_x(matrix); p_of_x = weyl_p(px_matrix)
    xp_matrix = weyl_p(matrix); x_of_p = weyl_x(xp_matrix)
    difference = p_of_x - x_of_p
    return float(np.linalg.norm(difference, 'fro'))

def discrete_wigner_function(matrix: NDArray) -> NDArray:
    """
    Calculates a simplified discrete Wigner function for a square matrix (state).
    W(q, p) = sum_y exp(-2*pi*i * p * y / N) <q+y | rho | q-y>
    """
    N = matrix.shape[0]
    wigner = np.zeros((N, N), dtype=float)
    for q in range(N):
        for p in range(N):
            val = 0j
            for y in range(N):
                idx1 = (q + y) % N
                idx2 = (q - y) % N
                # Using matrix elements as a density-like representation
                term = matrix[idx1, q] * np.conj(matrix[idx2, q])
                val += term * np.exp(-2j * np.pi * p * y / N)
            wigner[q, p] = np.real(val)
    return wigner

##############################################
# Correction Strategies                      #
##############################################

class CorrectionStrategy(ABC):
    @abstractmethod
    def get_correction_matrix(self, matrix: NDArray, angle_deg: float,
                              weyl_x: Operator, weyl_p: Operator) -> NDArray:
        pass

    def __repr__(self) -> str:
        params = [f"{k}={v}" for k, v in self.__dict__.items() if hasattr(self, k) and not k.startswith('_')]
        return f"{self.__class__.__name__}({', '.join(params)})"


class LemmaBasedCorrectionStrategy(CorrectionStrategy):
    def __init__(self, correction_factor: float = 0.01) -> None:
        self.correction_factor = correction_factor
    def get_correction_matrix(self, matrix: NDArray, angle_deg: float, weyl_x: Operator, weyl_p: Operator) -> NDArray:
        Q = estimate_quantization_effect_modular(matrix, weyl_x, weyl_p)
        scale = 1.0 / (1.0 + self.correction_factor * Q)
        return np.eye(matrix.shape[0], dtype=complex) * scale


class WeylAlgebraBasedCorrectionStrategy(CorrectionStrategy):
    def __init__(self, correction_factor: float = 0.01, commutator_weight: float = 0.2) -> None:
        self.correction_factor = correction_factor
        self.commutator_weight = commutator_weight
    def get_correction_matrix(self, matrix: NDArray, angle_deg: float, weyl_x: Operator, weyl_p: Operator) -> NDArray:
        px = weyl_x(matrix); p_of_x = weyl_p(px)
        xp = weyl_p(matrix); x_of_p = weyl_x(xp)
        commutator = p_of_x - x_of_p
        Q = float(np.linalg.norm(commutator, 'fro'))
        angle_factor = np.sin(2 * np.radians(angle_deg % 90))
        basic_scale = 1.0 / (1.0 + self.correction_factor * Q * (1 + angle_factor))
        dim = matrix.shape[0]
        correction = np.eye(dim, dtype=complex) * basic_scale
        if Q > 0:
            correction += (commutator / Q) * (1 - basic_scale) * self.commutator_weight * angle_factor
        return correction

class SymplecticCorrectionStrategy(CorrectionStrategy):
    def __init__(self, correction_factor: float = 0.01, phase_weight: float = 0.5) -> None:
        self.correction_factor = correction_factor
        self.phase_weight = phase_weight
    def get_correction_matrix(self, matrix: NDArray, angle_deg: float, weyl_x: Operator, weyl_p: Operator) -> NDArray:
        px = weyl_x(matrix); p_of_x = weyl_p(px)
        xp = weyl_p(matrix); x_of_p = weyl_x(xp)
        commutator = p_of_x - x_of_p
        Q = float(np.linalg.norm(commutator, 'fro'))
        scale = 1.0 / (1.0 + self.correction_factor * Q)
        dim = matrix.shape[0]
        correction = np.eye(dim, dtype=complex) * scale
        if Q > 0:
            phase_corr = np.exp(-1j * self.phase_weight * np.angle(commutator + 1e-12))
            correction += phase_corr * (np.abs(commutator) / Q) * (1 - scale)
        return correction

class WignerCorrectionStrategy(CorrectionStrategy):
    """
    Correction based on Wigner function smoothness.
    Constructs a filter that suppresses high-frequency artifacts in phase space.
    """
    def __init__(self, smooth_factor: float = 0.1) -> None:
        self.smooth_factor = smooth_factor
    def get_correction_matrix(self, matrix: NDArray, angle_deg: float, weyl_x: Operator, weyl_p: Operator) -> NDArray:
        W = discrete_wigner_function(matrix)
        # Low-pass filter in Wigner space
        W_smooth = scipy.ndimage.gaussian_filter(W, sigma=self.smooth_factor * matrix.shape[0])
        # Heuristic: use Wigner intensity to scale identity
        scale = np.mean(W_smooth) / (np.mean(W) + 1e-12)
        return np.eye(matrix.shape[0], dtype=complex) * np.clip(scale, 0.5, 1.0)

##############################################
# Rotation and Application                   #
##############################################

def rotate_matrix_around_center(matrix: NDArray, angle_degrees: float) -> NDArray:
    return scipy.ndimage.rotate(matrix, angle=angle_degrees, reshape=False, mode='constant', cval=0.0, order=1)

def apply_corrected_rotation(matrix: NDArray, C: NDArray, angle_degrees: float) -> NDArray:
    return rotate_matrix_around_center(C @ matrix.astype(complex), angle_degrees)

##############################################
# Helper Functions                           #
##############################################

def create_shape_matrix(shape_type: str, dim: int) -> NDArray:
    mat = np.zeros((dim, dim), dtype=float); cx, cy = dim // 2, dim // 2
    if shape_type == 'square':
        s = max(1, dim // 3); start = cx - s // 2; end = start + s
        mat[start:end, start:end] = 1.0
    elif shape_type == 'circle':
        r2 = (dim / 4.0)**2; Y, X = np.ogrid[:dim, :dim]
        mat[(X - cx)**2 + (Y - cy)**2 <= r2] = 1.0
    elif shape_type == 'triangle':
        bw = dim // 4; ay = cy // 2
        for i in range(ay, dim):
            hw = int(bw * (i - ay) / (dim - 1 - ay))
            mat[i, cx - hw : cx + hw + 1] = 1.0
    elif shape_type == 'line':
        t = max(1, dim // 16); mat[cy-t//2:cy+t//2, dim//4:3*dim//4] = 1.0
    elif shape_type == 'cross':
        a = max(1, dim // 10); mat[cy-a//2:cy+a//2, :] = 1.0; mat[:, cx-a//2:cx+a//2] = 1.0
    return mat

def calculate_rotation_quality(original: NDArray, standard_rotated: NDArray, corrected_rotated: NDArray) -> Dict[str, float]:
    overlap_std = float(np.sum(np.abs(original) * np.abs(standard_rotated)))
    overlap_corr = float(np.sum(np.abs(original) * np.abs(corrected_rotated)))
    return {
        'overlap_standard': overlap_std,
        'overlap_corrected': overlap_corr,
        'improvement_ratio': overlap_corr / (overlap_std + 1e-10)
    }

def visualize_all(initial: NDArray, standard: NDArray, corrected: NDArray, strategy_name: str):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    orig_abs = np.abs(initial); std_abs = np.abs(standard); corr_abs = np.abs(corrected)
    axes[0, 0].imshow(orig_abs, cmap='gray'); axes[0, 0].set_title("Original")
    axes[0, 1].imshow(std_abs, cmap='gray'); axes[0, 1].set_title("Standard Rotation")
    axes[0, 2].imshow(corr_abs, cmap='gray'); axes[0, 2].set_title(f"Corrected ({strategy_name})")

    # Wigner functions
    axes[1, 0].imshow(discrete_wigner_function(initial), cmap='magma'); axes[1, 0].set_title("Wigner (Original)")
    axes[1, 1].imshow(discrete_wigner_function(standard), cmap='magma'); axes[1, 1].set_title("Wigner (Standard)")
    axes[1, 2].imshow(discrete_wigner_function(corrected), cmap='magma'); axes[1, 2].set_title("Wigner (Corrected)")

    for ax in axes.flatten(): ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    DIM = 32; MODULUS = 32; QUANT = 1; ANGLE = 45.0
    ps = ModularPhaseSpace(DIM, MODULUS); wx = ModularWeylXOperator(ps, QUANT); wp = ModularWeylPOperator(ps, QUANT)

    shape = 'triangle'
    mat = create_shape_matrix(shape, DIM)
    std_rot = rotate_matrix_around_center(mat, ANGLE)

    strategies = [
        LemmaBasedCorrectionStrategy(),
        WeylAlgebraBasedCorrectionStrategy(),
        SymplecticCorrectionStrategy(),
        WignerCorrectionStrategy()
    ]

    for strat in strategies:
        print(f"Testing {strat}...")
        C = strat.get_correction_matrix(mat, ANGLE, wx, wp)
        corr_rot = apply_corrected_rotation(mat, C, ANGLE)
        metrics = calculate_rotation_quality(mat, std_rot, corr_rot)
        print(f"  Improvement: {metrics['improvement_ratio']:.4f}")
        # visualize_all(mat, std_rot, corr_rot, str(strat))
