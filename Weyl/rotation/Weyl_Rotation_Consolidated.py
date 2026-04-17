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

    Attributes:
        dimension (int): The (square) matrix dimension.
        modulus (int): A positive integer representing the cyclic group size.
    """
    def __init__(self, dimension: int, modulus: int) -> None:
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("Dimension must be a positive integer.")
        if not isinstance(modulus, int) or modulus <= 0:
            raise ValueError("Modulus must be a positive integer.")
        self.dimension: int = dimension
        self.modulus: int = modulus

    def get_index(self, index: int) -> int:
        """Return the index reduced modulo the modulus."""
        return index % self.modulus

    def shift_index(self, index: int, shift: int) -> int:
        """Shift the index by `shift` (mod modulus)."""
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
        """Applies the operator to the given matrix."""
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
        if matrix.shape[0] != self.phase_space.dimension:
            raise ValueError(f"Matrix dim {matrix.shape[0]} != phase space dim {self.phase_space.dimension}.")
        return np.roll(matrix, shift=self.quantization_level, axis=0)


class ModularWeylPOperator(Operator):
    """
    Implements a discrete clock (phase) operator.
    P |j> = exp(2*pi*i * j * q / N) |j>

    This operator does NOT commute with the shift operator X.
    [P, X] != 0
    """
    def __init__(self, phase_space: ModularPhaseSpace, quantization_level: int = 1) -> None:
        super().__init__(phase_space)
        self.quantization_level: int = quantization_level % self.phase_space.modulus

        # Precompute phase vector
        N = self.phase_space.dimension
        j = np.arange(N)
        self.phases = np.exp(2j * np.pi * j * self.quantization_level / N).reshape(-1, 1)

    def operate(self, matrix: NDArray) -> NDArray:
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Matrix must be square.")
        if matrix.shape[0] != self.phase_space.dimension:
            raise ValueError(f"Matrix dim {matrix.shape[0]} != phase space dim {self.phase_space.dimension}.")

        # Apply phase shift to each column
        return matrix * self.phases


def estimate_quantization_effect_modular(matrix: NDArray, weyl_x: Operator, weyl_p: Operator) -> float:
    """
    Estimates the quantization (non-commutativity) effect.
    Q = || P(X(matrix)) - X(P(matrix)) ||_F
    """
    px_matrix = weyl_x(matrix)
    p_of_x = weyl_p(px_matrix)

    xp_matrix = weyl_p(matrix)
    x_of_p = weyl_x(xp_matrix)

    difference = p_of_x - x_of_p
    return float(np.linalg.norm(difference, 'fro'))


def enhanced_estimate_quantization_effect(matrix: NDArray, weyl_x: Operator, weyl_p: Operator,
                                         angle_deg: float) -> Dict[str, float]:
    """
    Enhanced estimation of quantization effects that accounts for rotation angle.
    """
    px_matrix = weyl_x(matrix)
    p_of_x = weyl_p(px_matrix)
    xp_matrix = weyl_p(matrix)
    x_of_p = weyl_x(xp_matrix)
    commutator = p_of_x - x_of_p
    norm_q = float(np.linalg.norm(commutator, 'fro'))

    # Angle dependency - non-commutativity matters more at certain angles
    angle_rad = np.radians(angle_deg % 90)
    angle_factor = np.sin(2 * angle_rad)  # Peak at 45°

    effective_q = norm_q * (1 + angle_factor)

    return {
        'commutator_norm': norm_q,
        'angle_factor': angle_factor,
        'effective_q': effective_q
    }

##############################################
# Correction Strategies                      #
##############################################

class CorrectionStrategy(ABC):
    """Abstract base class for correction strategies."""
    @abstractmethod
    def get_correction_matrix(self, matrix: NDArray, angle_deg: float,
                              weyl_x: Operator, weyl_p: Operator) -> NDArray:
        pass

    def __repr__(self) -> str:
        params = [f"{k}={v}" for k, v in self.__dict__.items()
                  if hasattr(self, k) and not k.startswith('_')]
        return f"{self.__class__.__name__}({', '.join(params)})"


class LemmaBasedCorrectionStrategy(CorrectionStrategy):
    """
    Scales identity matrix based on Q = ||PX - XP||.
    scale = 1.0 / (1.0 + factor * Q)
    """
    def __init__(self, correction_factor: float = 0.01) -> None:
        self.correction_factor = correction_factor
        self._last_q = None
        self._last_scale = None

    def get_correction_matrix(self, matrix: NDArray, angle_deg: float, weyl_x: Operator, weyl_p: Operator) -> NDArray:
        Q = estimate_quantization_effect_modular(matrix, weyl_x, weyl_p)
        scale = 1.0 / (1.0 + self.correction_factor * Q)
        self._last_q = Q
        self._last_scale = scale
        return np.eye(matrix.shape[0], dtype=float) * scale


class EnhancedLemmaBasedCorrectionStrategy(CorrectionStrategy):
    """
    Enhanced correction strategy accounting for both quantization effect and rotation angle.
    """
    def __init__(self, correction_factor: float = 0.01, angle_sensitivity: float = 0.5) -> None:
        self.correction_factor = correction_factor
        self.angle_sensitivity = angle_sensitivity
        self._last_q = None
        self._last_scale = None
        self._last_effective_q = None

    def get_correction_matrix(self, matrix: NDArray, angle_deg: float, weyl_x: Operator, weyl_p: Operator) -> NDArray:
        q_metrics = enhanced_estimate_quantization_effect(matrix, weyl_x, weyl_p, angle_deg)
        Q = q_metrics['commutator_norm']
        angle_factor = q_metrics['angle_factor']

        correction_strength = self.correction_factor * Q * (1 + self.angle_sensitivity * angle_factor)
        scale = 1.0 / (1.0 + correction_strength)

        self._last_q = Q
        self._last_effective_q = q_metrics['effective_q']
        self._last_scale = scale

        return np.eye(matrix.shape[0], dtype=float) * scale


class WeylAlgebraBasedCorrectionStrategy(CorrectionStrategy):
    """
    Advanced correction strategy using the commutator [X,P] to construct
    a non-uniform correction matrix.
    """
    def __init__(self, correction_factor: float = 0.01, commutator_weight: float = 0.2) -> None:
        self.correction_factor = correction_factor
        self.commutator_weight = commutator_weight
        self._last_q = None
        self._last_scale = None

    def get_correction_matrix(self, matrix: NDArray, angle_deg: float, weyl_x: Operator, weyl_p: Operator) -> NDArray:
        px_matrix = weyl_x(matrix)
        p_of_x = weyl_p(px_matrix)
        xp_matrix = weyl_p(matrix)
        x_of_p = weyl_x(xp_matrix)
        commutator = p_of_x - x_of_p

        Q = float(np.linalg.norm(commutator, 'fro'))
        self._last_q = Q

        angle_rad = np.radians(angle_deg % 90)
        angle_factor = np.sin(2 * angle_rad)

        basic_scale = 1.0 / (1.0 + self.correction_factor * Q * (1 + angle_factor))
        self._last_scale = basic_scale

        dim = matrix.shape[0]
        # We only take the real part of the commutator for the correction matrix
        # to keep the correction matrix real-valued for image processing.
        correction = np.eye(dim, dtype=float) * basic_scale

        if Q > 0:
            norm_commutator_real = np.real(commutator) / Q
            contribution_scale = (1 - basic_scale) * self.commutator_weight * angle_factor
            correction += norm_commutator_real * contribution_scale

        return correction


class RandomNoiseCorrectionStrategy(CorrectionStrategy):
    """Adds random noise to the identity matrix."""
    def __init__(self, noise_factor: float = 0.01) -> None:
        self.noise_factor = noise_factor

    def get_correction_matrix(self, matrix: NDArray, angle_deg: float, weyl_x: Operator, weyl_p: Operator) -> NDArray:
        dim = matrix.shape[0]
        noise = (np.random.rand(dim, dim) - 0.5) * self.noise_factor
        return np.eye(dim, dtype=float) + noise

##############################################
# Rotation and Application                   #
##############################################

def rotate_matrix_around_center(matrix: NDArray, angle_degrees: float) -> NDArray:
    """Rotates a 2D matrix around its center using scipy.ndimage.rotate."""
    return scipy.ndimage.rotate(matrix, angle=angle_degrees, reshape=False, mode='constant', cval=0.0, order=1)


def apply_corrected_rotation(matrix: NDArray, C: NDArray, angle_degrees: float) -> NDArray:
    """Applies correction C then rotates: Result = Rotate(C @ matrix)."""
    corrected_matrix = C @ matrix
    return rotate_matrix_around_center(corrected_matrix, angle_degrees)

##############################################
# Helper Functions                           #
##############################################

def create_shape_matrix(shape_type: str, dim: int) -> NDArray:
    """Creates a square matrix representing a simple geometric shape centered."""
    mat = np.zeros((dim, dim), dtype=float)
    cx, cy = dim // 2, dim // 2
    if shape_type == 'square':
        size = max(1, dim // 3)
        start = cx - size // 2; end = start + size
        mat[start:end, start:end] = 1.0
    elif shape_type == 'circle':
        radius_sq = (dim / 4.0)**2
        Y, X = np.ogrid[:dim, :dim]
        dist_sq = (X - cx)**2 + (Y - cy)**2
        mat[dist_sq <= radius_sq] = 1.0
    elif shape_type == 'triangle':
        base_half_width = dim // 4
        apex_y = cy // 2
        for i in range(apex_y, dim):
            progress = (i - apex_y) / (dim - 1 - apex_y)
            half_width = int(base_half_width * progress)
            mat[i, cx - half_width : cx + half_width + 1] = 1.0
    elif shape_type == 'line':
        thickness = max(1, dim // 16)
        mat[cy - thickness // 2 : cy + thickness // 2, dim // 4 : 3 * dim // 4] = 1.0
    elif shape_type == 'cross':
        arm = max(1, dim // 10)
        mat[cy - arm // 2 : cy + arm // 2, :] = 1.0
        mat[:, cx - arm // 2 : cx + arm // 2] = 1.0
    return mat


def calculate_rotation_quality(original: NDArray, standard_rotated: NDArray, corrected_rotated: NDArray) -> Dict[str, float]:
    """Calculates metrics: overlap and difference norm."""
    overlap_std = float(np.sum(original * standard_rotated))
    overlap_corr = float(np.sum(original * corrected_rotated))
    diff_norm = float(np.linalg.norm(corrected_rotated - standard_rotated, 'fro'))
    return {
        'overlap_standard': overlap_std,
        'overlap_corrected': overlap_corr,
        'difference_norm': diff_norm,
        'improvement_ratio': overlap_corr / (overlap_std + 1e-10)
    }

##############################################
# Visualization and Experiment Functions     #
##############################################

def display_weyl_commutation(matrix: NDArray, weyl_x: Operator, weyl_p: Operator, shape_name: str) -> plt.Figure:
    """Visualizes the non-commutativity of Weyl operators."""
    px_matrix = weyl_x(matrix); p_of_x = weyl_p(px_matrix)
    xp_matrix = weyl_p(matrix); x_of_p = weyl_x(xp_matrix)
    difference = p_of_x - x_of_p
    norm_q = float(np.linalg.norm(difference, 'fro'))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    # Using absolute value for visualization of complex results
    p_of_x_abs = np.abs(p_of_x)
    x_of_p_abs = np.abs(x_of_p)
    diff_abs = np.abs(difference)

    vmax_abs = max(p_of_x_abs.max(), x_of_p_abs.max(), 1e-6)
    vmax_diff = max(diff_abs.max(), 1e-6)

    axes[0].imshow(p_of_x_abs, cmap='gray', vmin=0, vmax=vmax_abs)
    axes[0].set_title("|P(X(matrix))|")

    axes[1].imshow(x_of_p_abs, cmap='gray', vmin=0, vmax=vmax_abs)
    axes[1].set_title("|X(P(matrix))|")

    im2 = axes[2].imshow(diff_abs, cmap='hot', vmin=0, vmax=vmax_diff)
    axes[2].set_title(f"|PX - XP|\nNorm = {norm_q:.3f}")

    for ax in axes: ax.axis('off')
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    fig.suptitle(f"Weyl Operator Non-Commutativity (Shape: {shape_name})", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    return fig


def display_matrices_graphical(original: NDArray, standard_rotated: NDArray, corrected_rotated: NDArray,
                               title_prefix: str, angle: float, quant_level: int,
                               strategy_info: str, modulus: int,
                               metrics: Dict[str, float]) -> plt.Figure:
    """Displays original, standard rotated, and corrected rotated matrices."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    vmax = max(original.max(), standard_rotated.max(), corrected_rotated.max(), 1e-6)

    axes[0].imshow(original, cmap='gray', vmin=0, vmax=vmax)
    axes[0].set_title("Original Shape")

    axes[1].imshow(standard_rotated, cmap='gray', vmin=0, vmax=vmax)
    axes[1].set_title(f"Standard Rotation\nOverlap: {metrics['overlap_standard']:.3f}")

    axes[2].imshow(corrected_rotated, cmap='gray', vmin=0, vmax=vmax)
    axes[2].set_title(f"Corrected ({strategy_info})\nOverlap: {metrics['overlap_corrected']:.3f}")

    for ax in axes: ax.axis('off')
    fig.suptitle(f"{title_prefix}\nAngle: {angle:.1f}°, Quant: {quant_level}, Mod: {modulus}", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.90])
    return fig


if __name__ == "__main__":
    # Example usage
    DIM = 64
    MODULUS = 64
    QUANT_LEVEL = 1
    ANGLE_DEG = 45.0

    phase_space = ModularPhaseSpace(DIM, MODULUS)
    weyl_x = ModularWeylXOperator(phase_space, QUANT_LEVEL)
    weyl_p = ModularWeylPOperator(phase_space, QUANT_LEVEL)

    shape = 'circle'
    initial_matrix = create_shape_matrix(shape, DIM)

    # 1. Show non-commutativity
    display_weyl_commutation(initial_matrix, weyl_x, weyl_p, shape)
    plt.show()

    # 2. Compare strategies
    strategies = [
        LemmaBasedCorrectionStrategy(correction_factor=0.01),
        EnhancedLemmaBasedCorrectionStrategy(correction_factor=0.01, angle_sensitivity=0.5),
        WeylAlgebraBasedCorrectionStrategy(correction_factor=0.01, commutator_weight=0.2)
    ]

    standard_rotated = rotate_matrix_around_center(initial_matrix, ANGLE_DEG)

    for strategy in strategies:
        correction_matrix = strategy.get_correction_matrix(initial_matrix, ANGLE_DEG, weyl_x, weyl_p)
        corrected_rotated = apply_corrected_rotation(initial_matrix, correction_matrix, ANGLE_DEG)
        metrics = calculate_rotation_quality(initial_matrix, standard_rotated, corrected_rotated)

        display_matrices_graphical(initial_matrix, standard_rotated, corrected_rotated,
                                  f"Strategy Comparison", ANGLE_DEG, QUANT_LEVEL,
                                  str(strategy), MODULUS, metrics)
        plt.show()
