import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from abc import ABC, abstractmethod
from typing import Dict

##############################################
# Core Mathematical Structures and Operators #
##############################################

class ModularPhaseSpace:
    """
    Represents a discrete phase space that is isomorphic to a cyclic group ℤ_modulus.

    Attributes:
        dimension (int): The (square) matrix dimension.
        modulus (int): A positive integer representing the cyclic group size.
        indices (np.ndarray): Array of indices reduced modulo `modulus`.
    """
    def __init__(self, dimension: int, modulus: int) -> None:
        if modulus <= 0:
            raise ValueError("Modulus must be a positive integer (defining the cyclic group Z_modulus).")
        self.dimension = dimension
        self.modulus = modulus
        self.indices = np.arange(dimension) % modulus

    def get_index(self, index: int) -> int:
        return index % self.modulus

    def shift_index(self, index: int, shift: int) -> int:
        return (index + shift) % self.modulus

    def get_modular_indices(self) -> np.ndarray:
        return self.indices

    def __repr__(self) -> str:
        return f"ModularPhaseSpace(dimension={self.dimension}, modulus={self.modulus})"

class Operator(ABC):
    def __init__(self, phase_space: ModularPhaseSpace) -> None:
        if not isinstance(phase_space, ModularPhaseSpace):
            raise TypeError("Operator must be initialized with a ModularPhaseSpace object.")
        self.phase_space = phase_space

    @abstractmethod
    def operate(self, matrix: np.ndarray) -> np.ndarray:
        pass

    def __call__(self, matrix: np.ndarray) -> np.ndarray:
        return self.operate(matrix)

class ModularWeylXOperator(Operator):
    def __init__(self, phase_space: ModularPhaseSpace, quantization_level: int = 1) -> None:
        super().__init__(phase_space)
        self.quantization_level = quantization_level

    def operate(self, matrix: np.ndarray) -> np.ndarray:
        if matrix.shape != (self.phase_space.dimension, self.phase_space.dimension):
            raise ValueError("Input must be a square NumPy matrix of specified dimension.")
        shifted_matrix = np.zeros_like(matrix)
        for i in range(self.phase_space.dimension):
            shifted_row_index = self.phase_space.shift_index(i, self.quantization_level)
            shifted_matrix[shifted_row_index, :] = matrix[i, :]
        return shifted_matrix

class ModularWeylPOperator(Operator):
    def __init__(self, phase_space: ModularPhaseSpace, quantization_level: int = 1) -> None:
        super().__init__(phase_space)
        self.quantization_level = quantization_level

    def operate(self, matrix: np.ndarray) -> np.ndarray:
        if matrix.shape != (self.phase_space.dimension, self.phase_space.dimension):
            raise ValueError("Input must be a square NumPy matrix of specified dimension.")
        shifted_matrix = np.zeros_like(matrix)
        for j in range(self.phase_space.dimension):
            shifted_col_index = self.phase_space.shift_index(j, self.quantization_level)
            shifted_matrix[:, shifted_col_index] = matrix[:, j]
        return shifted_matrix

def discrete_rotation_operator(angle_degrees: float, discrete_space_dimension: int) -> np.ndarray:
    angle_radians = np.radians(angle_degrees)
    rot2d = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians)],
        [np.sin(angle_radians),  np.cos(angle_radians)]
    ])
    rotation_matrix = np.eye(discrete_space_dimension)
    if discrete_space_dimension >= 2:
        rotation_matrix[0:2, 0:2] = rot2d
    return rotation_matrix

def apply_discrete_rotation(matrix: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
    if matrix.shape != rotation_matrix.shape:
        raise ValueError("Matrix and rotation matrix dimensions must match for discrete rotation.")
    return rotation_matrix @ matrix @ rotation_matrix.T

def estimate_quantization_effect_modular(matrix: np.ndarray,
                                         modular_weyl_x: Operator,
                                         modular_weyl_p: Operator) -> float:
    matrix_xp = modular_weyl_p(modular_weyl_x(matrix.copy()))
    matrix_px = modular_weyl_x(modular_weyl_p(matrix.copy()))
    return np.linalg.norm(matrix_xp - matrix_px, 'fro')

##############################################
# Correction Strategies (Abstract & Concrete) #
##############################################

class CorrectionStrategy(ABC):
    @abstractmethod
    def get_correction_matrix(self, initial_matrix: np.ndarray, rotation_angle_deg: float,
                              weyl_x: Operator, weyl_p: Operator) -> np.ndarray:
        pass

class LemmaBasedCorrectionStrategy(CorrectionStrategy):
    def __init__(self, correction_factor: float = 0.005) -> None:
        self.correction_factor = correction_factor

    def get_correction_matrix(self, initial_matrix: np.ndarray, rotation_angle_deg: float,
                              modular_weyl_x: Operator, modular_weyl_p: Operator) -> np.ndarray:
        q = estimate_quantization_effect_modular(initial_matrix, modular_weyl_x, modular_weyl_p)
        return np.eye(initial_matrix.shape[0]) * (1.0 - self.correction_factor * q)

class RandomNoiseCorrectionStrategy(CorrectionStrategy):
    def __init__(self, noise_factor: float = 0.01) -> None:
        self.noise_factor = noise_factor

    def get_correction_matrix(self, initial_matrix: np.ndarray, rotation_angle_deg: float,
                              modular_weyl_x: Operator, modular_weyl_p: Operator) -> np.ndarray:
        noise = np.random.rand(*initial_matrix.shape) - 0.5
        return np.eye(initial_matrix.shape[0]) + self.noise_factor * noise

def apply_modular_corrected_rotation(matrix: np.ndarray, rotation_matrix: np.ndarray,
                                     correction_matrix: np.ndarray) -> np.ndarray:
    if matrix.shape != rotation_matrix.shape or matrix.shape != correction_matrix.shape:
        raise ValueError("All matrices must have the same shape.")
    return rotation_matrix @ (correction_matrix @ matrix) @ rotation_matrix.T

##############################################
# Shape Creation & Rotation Quality Metrics  #
##############################################

def create_shape_matrix(shape_type: str, matrix_dim: int) -> np.ndarray:
    matrix = np.zeros((matrix_dim, matrix_dim), dtype=float)
    cx, cy = matrix_dim // 2, matrix_dim // 2

    if shape_type == 'square':
        s = matrix_dim // 3
        matrix[cx-s//2:cx+s//2, cy-s//2:cy+s//2] = 1
    elif shape_type == 'circle':
        for i in range(matrix_dim):
            for j in range(matrix_dim):
                if (i - cx)**2 + (j - cy)**2 <= (matrix_dim // 4)**2:
                    matrix[i, j] = 1
    elif shape_type == 'triangle':
        h = matrix_dim // 2
        b = matrix_dim // 3
        for i in range(matrix_dim):
            for j in range(matrix_dim):
                if i >= cy and i <= cy + h:
                    if abs(j - cx) <= b * (1 - (i - cy) / h) / 2:
                        matrix[i, j] = 1
    elif shape_type == 'line':
        lw = matrix_dim // 8
        matrix[cx - lw//2:cx + lw//2, matrix_dim//4:3*matrix_dim//4] = 1
    elif shape_type == 'cross':
        w = matrix_dim // 5
        matrix[cx - w//2:cx + w//2, :] = 1
        matrix[:, cy - w//2:cy + w//2] = 1

    return matrix

def calculate_rotation_quality(original: np.ndarray,
                               rotated_standard: np.ndarray,
                               rotated_corrected: np.ndarray) -> Dict[str, float]:
    overlap_standard = np.sum(original * rotated_standard)
    overlap_corrected = np.sum(original * rotated_corrected)
    diff_norm = np.linalg.norm(rotated_standard - rotated_corrected, ord='fro')
    return {
        'overlap_standard': overlap_standard,
        'overlap_corrected': overlap_corrected,
        'difference_norm': diff_norm
  }
