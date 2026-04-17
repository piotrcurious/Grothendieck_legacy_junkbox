import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union

##############################################
# Core Mathematical Structures and Operators #
##############################################

class ModularPhaseSpace:
    r"""
    Represents a discrete phase space that is isomorphic to a cyclic group \mathbb{Z}_{\text{modulus}}.
    
    Attributes:
        dimension (int): The (square) matrix dimension.
        modulus (int): A positive integer representing the cyclic group size.
        indices (np.ndarray): Array of indices reduced modulo `modulus`.
    """
    def __init__(self, dimension: int, modulus: int) -> None:
        if modulus <= 0:
            raise ValueError("Modulus must be a positive integer (defining the cyclic group Z_modulus).")
        self.dimension: int = dimension
        self.modulus: int = modulus
        self.indices: np.ndarray = np.arange(dimension) % modulus

    def get_index(self, index: int) -> int:
        """Return the index reduced modulo the modulus."""
        return index % self.modulus

    def shift_index(self, index: int, shift: int) -> int:
        """Shift the index by `shift` (mod modulus)."""
        return (index + shift) % self.modulus

    def get_modular_indices(self) -> np.ndarray:
        """Return all indices modulo the modulus."""
        return self.indices

    def __repr__(self) -> str:
        return f"ModularPhaseSpace(dimension={self.dimension}, modulus={self.modulus})"


class Operator(ABC):
    r"""
    Abstract base class representing an operator on a modular phase space.
    
    Subclasses must implement the `operate` method.
    """
    def __init__(self, phase_space: ModularPhaseSpace) -> None:
        if not isinstance(phase_space, ModularPhaseSpace):
            raise TypeError("Operator must be initialized with a ModularPhaseSpace object.")
        self.phase_space: ModularPhaseSpace = phase_space

    @abstractmethod
    def operate(self, matrix: np.ndarray) -> np.ndarray:
        r"""
        Applies the operator to the given matrix.
        
        Args:
            matrix (np.ndarray): A square matrix of shape (dimension, dimension).
            
        Returns:
            np.ndarray: The resulting matrix after applying the operator.
        """
        pass

    def __call__(self, matrix: np.ndarray) -> np.ndarray:
        return self.operate(matrix)


class ModularWeylXOperator(Operator):
    r"""
    Implements a discrete translation (shift) operator in the X direction (analogous to a Weyl operator).
    
    The operator shifts the rows of the matrix by an amount defined by `quantization_level`.
    """
    def __init__(self, phase_space: ModularPhaseSpace, quantization_level: int = 1) -> None:
        super().__init__(phase_space)
        self.quantization_level: int = quantization_level

    def operate(self, matrix: np.ndarray) -> np.ndarray:
        if not (isinstance(matrix, np.ndarray) and matrix.shape == (self.phase_space.dimension, self.phase_space.dimension)):
            raise ValueError("Input must be a square NumPy matrix of specified dimension.")
        shifted_matrix: np.ndarray = np.zeros_like(matrix)
        for i in range(self.phase_space.dimension):
            shifted_row_index = self.phase_space.shift_index(i, self.quantization_level)
            shifted_matrix[shifted_row_index, :] = matrix[i, :]
        return shifted_matrix


class ModularWeylPOperator(Operator):
    r"""
    Implements a discrete translation (shift) operator in the P direction (the conjugate direction).
    
    The operator shifts the columns of the matrix by an amount defined by `quantization_level`.
    """
    def __init__(self, phase_space: ModularPhaseSpace, quantization_level: int = 1) -> None:
        super().__init__(phase_space)
        self.quantization_level: int = quantization_level

    def operate(self, matrix: np.ndarray) -> np.ndarray:
        if not (isinstance(matrix, np.ndarray) and matrix.shape == (self.phase_space.dimension, self.phase_space.dimension)):
            raise ValueError("Input must be a square NumPy matrix of specified dimension.")
        shifted_matrix: np.ndarray = np.zeros_like(matrix)
        for j in range(self.phase_space.dimension):
            shifted_col_index = self.phase_space.shift_index(j, self.quantization_level)
            shifted_matrix[:, shifted_col_index] = matrix[:, j]
        return shifted_matrix


def discrete_rotation_operator(angle_degrees: float, discrete_space_dimension: int) -> np.ndarray:
    r"""
    Constructs a discrete rotation matrix by embedding a continuous 2D rotation into a larger space.
    
    The rotation acts nontrivially on the first two coordinates and leaves remaining directions invariant.
    
    Args:
        angle_degrees (float): Rotation angle in degrees.
        discrete_space_dimension (int): Dimension of the discrete space.
        
    Returns:
        np.ndarray: A (discrete_space_dimension x discrete_space_dimension) rotation matrix.
    """
    angle_radians = np.radians(angle_degrees)
    continuous_rotation_matrix_2d = np.array([[np.cos(angle_radians), -np.sin(angle_radians)],
                                               [np.sin(angle_radians),  np.cos(angle_radians)]])
    discrete_rotation_matrix = np.eye(discrete_space_dimension, dtype=float)
    if discrete_space_dimension >= 2:
        discrete_rotation_matrix[0:2, 0:2] = continuous_rotation_matrix_2d
    return discrete_rotation_matrix


def apply_discrete_rotation(matrix: np.ndarray, discrete_rotation_mat: np.ndarray) -> np.ndarray:
    r"""
    Applies a discrete rotation to a matrix.
    
    The rotation is performed by a similarity transform:
    
        M_{\text{rot}} = R \, M \, R^T
    
    Args:
        matrix (np.ndarray): The matrix to be rotated.
        discrete_rotation_mat (np.ndarray): The rotation matrix.
        
    Returns:
        np.ndarray: The rotated matrix.
    """
    if matrix.shape != discrete_rotation_mat.shape:
        raise ValueError("Matrix and rotation matrix dimensions must match for discrete rotation.")
    rotated_matrix: np.ndarray = discrete_rotation_mat @ matrix @ discrete_rotation_mat.T
    return rotated_matrix


def estimate_quantization_effect_modular(matrix: np.ndarray,
                                         modular_weyl_x: Operator,
                                         modular_weyl_p: Operator) -> float:
    r"""
    Estimates the quantization (non-commutativity) effect inherent to the modular Weyl operators.
    
    This is done by computing the Frobenius norm of the difference between the two compositions:
    
        \Delta = \|\,W_P(W_X(M)) - W_X(W_P(M))\,\|_F
    
    Args:
        matrix (np.ndarray): The input matrix.
        modular_weyl_x (Operator): The Weyl-X operator.
        modular_weyl_p (Operator): The Weyl-P operator.
        
    Returns:
        float: The estimated quantization effect.
    """
    matrix_xp = modular_weyl_p(modular_weyl_x(matrix.copy()))
    matrix_px = modular_weyl_x(modular_weyl_p(matrix.copy()))
    difference_matrix = matrix_xp - matrix_px
    quantization_estimate = np.linalg.norm(difference_matrix, 'fro')
    return quantization_estimate

##############################################
# Correction Strategies (Abstract & Concrete) #
##############################################

class CorrectionStrategy(ABC):
    r"""
    Abstract base class for defining correction strategies.
    
    A correction strategy is used to compute a correction matrix that aims to mitigate
    the quantization effects of applying modular Weyl operators.
    """
    @abstractmethod
    def get_correction_matrix(self, initial_matrix: np.ndarray, rotation_angle_deg: float,
                              weyl_x: Operator, weyl_p: Operator) -> np.ndarray:
        """
        Computes and returns a correction matrix.
        """
        pass


class LemmaBasedCorrectionStrategy(CorrectionStrategy):
    r"""
    Correction strategy based on a lemma that suggests scaling the identity matrix
    by a factor related to the quantization estimate.
    
    The correction matrix is given by:
    
        C = (1 - \epsilon \cdot Q) I
    
    where Q is the quantization estimate and \epsilon is a small correction factor.
    """
    def __init__(self, correction_factor: float = 0.005) -> None:
        self.correction_factor: float = correction_factor

    def get_correction_matrix(self, initial_matrix: np.ndarray, rotation_angle_deg: float,
                              modular_weyl_x: Operator, modular_weyl_p: Operator) -> np.ndarray:
        quantization_estimate = estimate_quantization_effect_modular(initial_matrix, modular_weyl_x, modular_weyl_p)
        correction_scale = self.correction_factor * quantization_estimate
        correction_matrix = np.eye(initial_matrix.shape[0]) * (1.0 - correction_scale)
        return correction_matrix


class RandomNoiseCorrectionStrategy(CorrectionStrategy):
    r"""
    A baseline correction strategy that adds small random noise to the identity.
    
    This strategy is primarily intended as a control to compare against the lemma-based correction.
    """
    def __init__(self, noise_factor: float = 0.01) -> None:
        self.noise_factor: float = noise_factor

    def get_correction_matrix(self, initial_matrix: np.ndarray, rotation_angle_deg: float,
                              modular_weyl_x: Operator, modular_weyl_p: Operator) -> np.ndarray:
        noise_matrix = np.random.rand(*initial_matrix.shape) - 0.5  # Values in [-0.5, 0.5]
        correction_matrix = np.eye(initial_matrix.shape[0]) + self.noise_factor * noise_matrix
        return correction_matrix


def apply_modular_corrected_rotation(matrix: np.ndarray, discrete_rotation_mat: np.ndarray,
                                     correction_matrix: np.ndarray) -> np.ndarray:
    r"""
    Applies a modular-corrected rotation to the matrix.
    
    First, a correction matrix is applied to the original matrix, and then the discrete rotation
    is performed:
    
        M_{\text{corrected}} = R \, (C\,M) \, R^T
    
    Args:
        matrix (np.ndarray): The original matrix.
        discrete_rotation_mat (np.ndarray): The rotation matrix.
        correction_matrix (np.ndarray): The correction matrix.
        
    Returns:
        np.ndarray: The corrected, rotated matrix.
    """
    if matrix.shape != discrete_rotation_mat.shape or matrix.shape != correction_matrix.shape:
        raise ValueError("Matrix and rotation/correction matrices must have compatible dimensions.")
    corrected_matrix_pre_rotation = correction_matrix @ matrix
    rotated_matrix_corrected = discrete_rotation_mat @ corrected_matrix_pre_rotation @ discrete_rotation_mat.T
    return rotated_matrix_corrected

##############################################
# Helper Functions for Shape Creation & Quality#
##############################################

def create_shape_matrix(shape_type: str, matrix_dim: int) -> np.ndarray:
    r"""
    Generates a matrix representation of a geometric shape.
    
    The matrix is of size (matrix_dim x matrix_dim) and contains 1’s where the shape is present,
    and 0’s elsewhere.
    
    Supported shapes:
        - 'square'
        - 'circle'
        - 'triangle'
        - 'line'
        - 'cross'
    
    Args:
        shape_type (str): The type of shape.
        matrix_dim (int): The matrix dimension.
        
    Returns:
        np.ndarray: The binary matrix representing the shape.
    """
    matrix = np.zeros((matrix_dim, matrix_dim), dtype=float)
    center_x, center_y = matrix_dim // 2, matrix_dim // 2

    if shape_type == 'square':
        size = matrix_dim // 3
        start = matrix_dim // 2 - size // 2
        end = start + size
        matrix[start:end, start:end] = 1

    elif shape_type == 'circle':
        radius = matrix_dim // 4
        for i in range(matrix_dim):
            for j in range(matrix_dim):
                if (i - center_x)**2 + (j - center_y)**2 <= radius**2:
                    matrix[i, j] = 1

    elif shape_type == 'triangle':  # Rudimentary triangle approximation
        base_width = matrix_dim // 3
        height = matrix_dim // 2
        for i in range(matrix_dim):
            for j in range(matrix_dim):
                if i >= center_y and i <= center_y + height and abs(j - center_x) <= base_width * (1 - (i - center_y) / height) / 2:
                    matrix[i, j] = 1

    elif shape_type == 'line':  # Horizontal line
        line_width = matrix_dim // 8
        start_x = matrix_dim // 4
        end_x = 3 * matrix_dim // 4
        center_row = matrix_dim // 2
        matrix[center_row - line_width // 2: center_row + line_width // 2, start_x:end_x] = 1

    elif shape_type == 'cross':  # Simple cross shape
        cross_width = matrix_dim // 5
        center_row, center_col = matrix_dim // 2, matrix_dim // 2
        matrix[center_row - cross_width // 2: center_row + cross_width // 2, :] = 1  # Vertical bar
        matrix[:, center_col - cross_width // 2: center_col + cross_width // 2] = 1  # Horizontal bar

    return matrix


def calculate_rotation_quality(original_shape: np.ndarray,
                               rotated_shape_standard: np.ndarray,
                               rotated_shape_corrected: np.ndarray) -> Dict[str, float]:
    r"""
    Computes heuristic metrics to assess the quality of a rotation.
    
    The metrics include:
      1. The overlap between the original shape and the rotated shape.
      2. The Frobenius norm of the difference between the standard and corrected rotations.
    
    Args:
        original_shape (np.ndarray): The original binary shape matrix.
        rotated_shape_standard (np.ndarray): The matrix after standard rotation.
        rotated_shape_corrected (np.ndarray): The matrix after applying the correction.
        
    Returns:
        Dict[str, float]: A dictionary containing:
            - 'overlap_standard': Sum of element-wise product of original and standard.
            - 'overlap_corrected': Sum of element-wise product of original and corrected.
            - 'difference_norm': Frobenius norm of (corrected - standard).
    """
    overlap_standard = float(np.sum(original_shape * rotated_shape_standard))
    overlap_corrected = float(np.sum(original_shape * rotated_shape_corrected))
    difference_matrix = rotated_shape_corrected - rotated_shape_standard
    difference_norm = float(np.linalg.norm(difference_matrix, 'fro'))
    return {
        'overlap_standard': overlap_standard,
        'overlap_corrected': overlap_corrected,
        'difference_norm': difference_norm,
    }

##############################################
# Visualization and Experiment Functions     #
##############################################

def display_matrices_graphical(matrix_standard: np.ndarray, matrix_corrected: np.ndarray,
                               angle: float, quantization: int, correction_factor: float,
                               modulus: int, quality_metrics: Optional[Dict[str, float]] = None,
                               correction_strategy_name: str = "Lemma-Based") -> Any:
    """
    Displays the standard and corrected rotated matrices side by side along with quality metrics.
    
    Args:
        matrix_standard (np.ndarray): Matrix from standard discrete rotation.
        matrix_corrected (np.ndarray): Matrix from corrected rotation.
        angle (float): Rotation angle in degrees.
        quantization (int): Current quantization level.
        correction_factor (float): The correction (or noise) factor.
        modulus (int): The modulus value.
        quality_metrics (Optional[Dict[str, float]]): Dictionary of quality metrics.
        correction_strategy_name (str): Name of the correction strategy.
        
    Returns:
        Tuple containing the figure, axes, and image objects.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    im1 = axes[0].imshow(matrix_standard, cmap='gray', interpolation='nearest')
    title_standard = 'Standard Discrete Rotation'
    if quality_metrics:
        title_standard += f"\nOverlap: {quality_metrics['overlap_standard']:.2f}"
    axes[0].set_title(title_standard)
    axes[0].axis('off')

    im2 = axes[1].imshow(matrix_corrected, cmap='gray', interpolation='nearest')
    title_corrected = f'Corrected Rotation ({correction_strategy_name})'
    if quality_metrics:
        title_corrected += f"\nOverlap: {quality_metrics['overlap_corrected']:.2f}, Diff Norm: {quality_metrics['difference_norm']:.2f}"
    axes[1].set_title(title_corrected)
    axes[1].axis('off')

    fig.suptitle(f'Rotation Angle: {angle}° | Quantization Level: {quantization} | '
                 f'Correction Factor: {correction_factor:.4f} | Modulus: {modulus}')
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    return fig, axes, [im1, im2]

demonstrate_modular_weyl_algebra_and_rotation_graphical(matrix_dimension_graphical,
                                                            modulus_value_graphical,
                                                            quantization_initial,
                                                            rotation_angle_initial,
                                                            correction_factor_initial,
                                                            shape_type=shape,
                                                            correction_strategy_type=correction_strategy_type_initial)

    # Run experiment varying quantization level
    experiment_quantization_levels = [1, 2, 3, 4, 5]
    lemma_correction = LemmaBasedCorrectionStrategy(correction_factor=0.005)
    experiment_results_quantization = run_experiment_parameter_variation(matrix_dimension_graphical,
                                                                         modulus_value_graphical,
                                                                         fixed_quantization_level=quantization_initial,
                                                                         fixed_rotation_angle_deg=rotation_angle_initial,
                                                                         correction_strategy=lemma_correction,
                                                                         param_name='quantization_level',
                                                                         param_values=experiment_quantization_levels,
                                                                         shape_type=shape)
    plot_experiment_results(experiment_results_quantization, 'quantization_level',
                            modulus_value_graphical, "Lemma-Based")

    # Run experiment varying rotation angle
    experiment_angles = np.linspace(0, 180, 10)
    experiment_results_angles = run_experiment_parameter_variation(matrix_dimension_graphical,
                                                                   modulus_value_graphical,
                                                                   fixed_quantization_level=quantization_initial,
                                                                   fixed_rotation_angle_deg=rotation_angle_initial,
                                                                   correction_strategy=lemma_correction,
                                                                   param_name='rotation_angle_deg',
                                                                   param_values=list(experiment_angles),
                                                                   shape_type=shape)
    plot_experiment_results(experiment_results_angles, 'rotation_angle_deg',
                            modulus_value_graphical, "Lemma-Based")

    # Example with Random Noise Correction Experiment
    random_noise_correction = RandomNoiseCorrectionStrategy(noise_factor=0.01)
    experiment_results_quantization_noise = run_experiment_parameter_variation(matrix_dimension_graphical,
                                                                               modulus_value_graphical,
                                                                               fixed_quantization_level=quantization_initial,
                                                                               fixed_rotation_angle_deg=rotation_angle_initial,
                                                                               correction_strategy=random_noise_correction,
                                                                               param_name='quantization_level',
                                                                               param_values=experiment_quantization_levels,
                                                                               shape_type=shape)
    plot_experiment_results(experiment_results_quantization_noise, 'quantization_level',
                            modulus_value_graphical, "Random Noise")
