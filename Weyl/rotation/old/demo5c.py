import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Type
# For more specific numpy type hinting (optional, requires recent numpy)
# from numpy.typing import NDArray
# Otherwise, use np.ndarray
NDArray = np.ndarray # Alias for convenience if not using numpy.typing

##############################################
# Core Mathematical Structures and Operators
##############################################

class ModularPhaseSpace:
    """
    Represents a discrete phase space isomorphic to Z_modulus.

    Provides methods for modular arithmetic on indices within a defined dimension
    and modulus.
    """
    def __init__(self, dimension: int, modulus: int) -> None:
        """
        Initializes the modular phase space.

        Args:
            dimension: The dimension of the phase space (number of points).
            modulus: The modulus for the cyclic group Z_modulus.
        """
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("Dimension must be a positive integer.")
        if not isinstance(modulus, int) or modulus <= 0:
            raise ValueError("Modulus must be a positive integer.")
            
        self.dimension = dimension
        self.modulus = modulus
        # self.indices = np.arange(dimension) % modulus # Not strictly needed for current methods

    def get_index(self, index: int) -> int:
        """
        Applies the modulus to an index.

        Args:
            index: The raw index.

        Returns:
            The index modulo the phase space modulus.
        """
        return index % self.modulus

    def shift_index(self, index: int, shift: int) -> int:
        """
        Shifts an index within the modular phase space.

        Args:
            index: The starting index.
            shift: The amount to shift by.

        Returns:
            The new index after modular addition.
        """
        return (index + shift) % self.modulus

    def __repr__(self) -> str:
        """Provides a string representation of the object."""
        return f"ModularPhaseSpace(dimension={self.dimension}, modulus={self.modulus})"

class Operator(ABC):
    """
    Abstract base class for operators acting on matrices in a ModularPhaseSpace.
    """
    def __init__(self, phase_space: ModularPhaseSpace) -> None:
        """
        Initializes the operator with its phase space.

        Args:
            phase_space: The modular phase space the operator acts upon.

        Raises:
            TypeError: If phase_space is not a ModularPhaseSpace instance.
        """
        if not isinstance(phase_space, ModularPhaseSpace):
            raise TypeError("Operator must be initialized with a ModularPhaseSpace.")
        self.phase_space = phase_space

    @abstractmethod
    def operate(self, matrix: NDArray) -> NDArray:
        """
        Applies the operator to a matrix. Abstract method.

        Args:
            matrix: The input matrix (typically a square NumPy array).

        Returns:
            The resulting matrix after the operator is applied.
        """
        pass

    def __call__(self, matrix: NDArray) -> NDArray:
        """Allows the operator instance to be called like a function."""
        return self.operate(matrix)

class ModularWeylXOperator(Operator):
    """
    Represents the modular Weyl X (position shift) operator.
    Shifts matrix rows downwards (with wrap-around).
    """
    def __init__(self, phase_space: ModularPhaseSpace, quantization_level: int = 1) -> None:
        """
        Initializes the Weyl X operator.

        Args:
            phase_space: The modular phase space.
            quantization_level: The amount of shift to apply (modulo phase_space.modulus).
        """
        super().__init__(phase_space)
        self.quantization_level = quantization_level % self.phase_space.modulus

    def operate(self, matrix: NDArray) -> NDArray:
        """
        Applies the X operator (row shift) to the matrix.

        Args:
            matrix: The input square matrix.

        Returns:
            The matrix with rows shifted by quantization_level.

        Raises:
            ValueError: If the matrix is not square or dimension doesn't match phase space.
        """
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
             raise ValueError("Matrix must be square.")
        if matrix.shape[0] != self.phase_space.dimension:
            raise ValueError(f"Matrix dimension {matrix.shape[0]} must match phase space dimension {self.phase_space.dimension}.")

        # Use np.roll for efficient circular shift along axis 0 (rows)
        # Positive shift in np.roll moves elements "down" (higher indices)
        return np.roll(matrix, shift=self.quantization_level, axis=0)

class ModularWeylPOperator(Operator):
    """
    Represents the modular Weyl P (momentum shift) operator.
    Shifts matrix columns rightwards (with wrap-around).
    Note: The relation to actual momentum depends on the representation (e.g., Fourier).
    """
    def __init__(self, phase_space: ModularPhaseSpace, quantization_level: int = 1) -> None:
        """
        Initializes the Weyl P operator.

        Args:
            phase_space: The modular phase space.
            quantization_level: The amount of shift to apply (modulo phase_space.modulus).
        """
        super().__init__(phase_space)
        self.quantization_level = quantization_level % self.phase_space.modulus

    def operate(self, matrix: NDArray) -> NDArray:
        """
        Applies the P operator (column shift) to the matrix.

        Args:
            matrix: The input square matrix.

        Returns:
            The matrix with columns shifted by quantization_level.

        Raises:
            ValueError: If the matrix is not square or dimension doesn't match phase space.
        """
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
             raise ValueError("Matrix must be square.")
        if matrix.shape[0] != self.phase_space.dimension:
            raise ValueError(f"Matrix dimension {matrix.shape[0]} must match phase space dimension {self.phase_space.dimension}.")

        # Use np.roll for efficient circular shift along axis 1 (columns)
        # Positive shift in np.roll moves elements "right" (higher indices)
        return np.roll(matrix, shift=self.quantization_level, axis=1)

def discrete_rotation_operator(angle_degrees: float, dim: int) -> NDArray:
    """
    Creates a rotation matrix that primarily rotates the first two dimensions.

    Note: This implementation applies a standard 2D rotation to the subspace
    spanned by the first two basis vectors and leaves other dimensions unchanged.
    It might not represent a general N-dimensional rotation.

    Args:
        angle_degrees: The rotation angle in degrees.
        dim: The dimension of the square matrix.

    Returns:
        A (dim x dim) rotation matrix.
    """
    theta = np.radians(angle_degrees)
    # 2D rotation matrix
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    R2 = np.array([[cos_theta, -sin_theta],
                   [sin_theta,  cos_theta]], dtype=float)

    # Embed in a dim x dim identity matrix
    R = np.eye(dim, dtype=float)
    if dim >= 2:
        R[0:2, 0:2] = R2
    return R

def apply_discrete_rotation(matrix: NDArray, R: NDArray) -> NDArray:
    """
    Applies a rotation to a matrix using R @ matrix @ R.T.

    Args:
        matrix: The matrix to rotate (dim x dim).
        R: The rotation matrix (dim x dim).

    Returns:
        The rotated matrix.

    Raises:
        ValueError: If matrix and R shapes are incompatible.
    """
    if matrix.shape != R.shape or matrix.ndim != 2 or R.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Shapes must match and be square 2D arrays. Got matrix: {matrix.shape}, R: {R.shape}")
    # R.T is the transpose, equivalent to R inverse for rotation matrices
    return R @ matrix @ R.T

def estimate_quantization_effect_modular(matrix: NDArray, weyl_x: Operator, weyl_p: Operator) -> float:
    """
    Estimates the effect of quantization by calculating the Frobenius norm
    of the commutator [P, X] applied to the matrix.
    || P(X(matrix)) - X(P(matrix)) ||_F

    Args:
        matrix: The input matrix.
        weyl_x: The ModularWeylXOperator instance.
        weyl_p: The ModularWeylPOperator instance.

    Returns:
        The Frobenius norm of the difference matrix, quantifying the non-commutativity effect.
    """
    # No need for matrix.copy() as weyl operators return new arrays
    xp = weyl_p(weyl_x(matrix))
    px = weyl_x(weyl_p(matrix))
    # Frobenius norm
    return float(np.linalg.norm(xp - px, 'fro'))

##############################################
# Correction Strategies
##############################################

class CorrectionStrategy(ABC):
    """Abstract base class for defining strategies to correct quantization effects."""
    @abstractmethod
    def get_correction_matrix(self, matrix: NDArray, angle_deg: float,
                              weyl_x: Operator, weyl_p: Operator) -> NDArray:
        """
        Calculates a correction matrix based on the strategy.

        Args:
            matrix: The original matrix (before rotation).
            angle_deg: The intended rotation angle in degrees.
            weyl_x: The Weyl X operator instance.
            weyl_p: The Weyl P operator instance.

        Returns:
            A correction matrix C (usually dim x dim).
        """
        pass

class LemmaBasedCorrectionStrategy(CorrectionStrategy):
    """
    A correction strategy based on scaling the identity matrix.
    The scaling factor depends on the estimated quantization effect.
    """
    def __init__(self, correction_factor: float = 0.005, epsilon: float = 1e-9) -> None:
        """
        Initializes the lemma-based correction strategy.

        Args:
            correction_factor: A factor controlling the strength of the correction.
            epsilon: A small value to prevent division by zero or instability.
        """
        if correction_factor < 0:
             print(f"Warning: Negative correction_factor ({correction_factor}) may lead to unexpected results.")
        self.correction_factor = correction_factor
        self.epsilon = epsilon # To avoid potential issues if Q is huge, though unlikely

    def get_correction_matrix(self, matrix: NDArray, angle_deg: float,
                              weyl_x: Operator, weyl_p: Operator) -> NDArray:
        """
        Calculates a scaled identity matrix as the correction matrix.

        Args:
            matrix: The original matrix.
            angle_deg: The rotation angle (unused in this simple strategy).
            weyl_x: The Weyl X operator.
            weyl_p: The Weyl P operator.

        Returns:
            A scaled identity matrix (dim x dim).
        """
        Q = estimate_quantization_effect_modular(matrix, weyl_x, weyl_p)
        # Scale factor based on quantization effect
        # Adding epsilon for numerical stability, though Q is norm >= 0
        scale = 1.0 - self.correction_factor * Q # Removed 1/(1+...) for simplicity, closer to original
        # Ensure scale is not excessively negative if correction_factor * Q > 1
        scale = max(scale, self.epsilon) # Prevent negative or zero scaling

        return np.eye(matrix.shape[0], dtype=float) * scale

class RandomNoiseCorrectionStrategy(CorrectionStrategy):
    """
    A correction strategy that adds random noise to the identity matrix.
    """
    def __init__(self, noise_factor: float = 0.01) -> None:
        """
        Initializes the random noise correction strategy.

        Args:
            noise_factor: Controls the magnitude of the random noise added.
        """
        self.noise_factor = noise_factor

    def get_correction_matrix(self, matrix: NDArray, angle_deg: float,
                              weyl_x: Operator, weyl_p: Operator) -> NDArray:
        """
        Generates a correction matrix by adding scaled random noise to identity.

        Args:
            matrix: The original matrix (used for shape).
            angle_deg: The rotation angle (unused).
            weyl_x: The Weyl X operator (unused).
            weyl_p: The Weyl P operator (unused).

        Returns:
            An identity matrix plus scaled random noise (dim x dim).
        """
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
             raise ValueError("Matrix must be square 2D.")
        dim = matrix.shape[0]
        # Uniform noise in [-0.5 * noise_factor, 0.5 * noise_factor]
        noise = (np.random.rand(dim, dim) - 0.5) * self.noise_factor
        return np.eye(dim, dtype=float) + noise

def apply_modular_corrected_rotation(matrix: NDArray, R: NDArray, C: NDArray) -> NDArray:
    """
    Applies a corrected rotation: R @ (C @ matrix) @ R.T.

    Args:
        matrix: The original matrix (dim x dim).
        R: The rotation matrix (dim x dim).
        C: The correction matrix (dim x dim).

    Returns:
        The matrix after applying correction and rotation.

    Raises:
        ValueError: If matrix, R, and C shapes are incompatible.
    """
    if not (matrix.shape == R.shape == C.shape):
        raise ValueError(f"Shapes must match. Got matrix: {matrix.shape}, R: {R.shape}, C: {C.shape}")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Inputs must be square 2D arrays.")

    # Apply correction C first, then rotation R
    corrected_matrix = C @ matrix
    rotated_corrected_matrix = R @ corrected_matrix @ R.T
    return rotated_corrected_matrix

##############################################
# Helper Functions
##############################################

def create_shape_matrix(shape_type: str, dim: int) -> NDArray:
    """
    Creates a square matrix representing a geometric shape.

    Args:
        shape_type: Type of shape ('square', 'circle', 'triangle', 'line', 'cross').
        dim: The dimension of the square matrix.

    Returns:
        A (dim x dim) NumPy array with 1.0s for the shape and 0.0s elsewhere.

    Raises:
        ValueError: If shape_type is not supported or dim is not positive.
    """
    if dim <= 0:
        raise ValueError("Dimension must be positive.")
        
    mat = np.zeros((dim, dim), dtype=float)
    cx, cy = dim // 2, dim // 2 # Center coordinates

    if shape_type == 'square':
        size = dim // 3 # Size of the square side
        start = cx - size // 2
        end = start + size
        mat[start:end, start:end] = 1.0
    elif shape_type == 'circle':
        radius_sq = (dim / 4.0)**2
        Y, X = np.ogrid[:dim, :dim] # Note: ogrid gives Y, X
        # Create mask for pixels within the circle radius
        mask = (X - cx)**2 + (Y - cy)**2 <= radius_sq
        mat[mask] = 1.0
    elif shape_type == 'triangle':
        # Creates an isosceles triangle pointing down from the center top
        # This implementation might not be perfectly centered or symmetrical
        # depending on rounding for odd/even dim.
        for i in range(cy): # Iterate rows from top to center
             half_width = int(i * (cx / cy)) if cy > 0 else 0
             j_start = cx - half_width
             j_end = cx + half_width + 1 # +1 for inclusive slicing end
             mat[i, max(0, j_start):min(dim, j_end)] = 1.0
        # Alternate triangle implementation (similar to original):
        # for i in range(cx, dim):
        #     width = (dim - 1 - i) * cx / (dim - 1 - cx) if (dim - 1 - cx) != 0 else 0
        #     # Ensure width doesn't exceed bounds
        #     width = min(width, cx)
        #     j0 = int(cx - width)
        #     j1 = int(cx + width) + 1
        #     mat[i, max(0, j0):min(dim, j1)] = 1.0

    elif shape_type == 'line':
        # Horizontal line across the middle
        line_width = max(1, dim // 16) # Thinner line
        row_start = cx - line_width // 2
        row_end = row_start + line_width
        col_start = dim // 4
        col_end = 3 * dim // 4
        mat[max(0, row_start):min(dim, row_end), max(0, col_start):min(dim, col_end)] = 1.0
    elif shape_type == 'cross':
        # Centered cross shape
        arm_width = max(1, dim // 10) # Width of the arms
        # Horizontal arm
        row_start_h = cx - arm_width // 2
        row_end_h = row_start_h + arm_width
        mat[max(0, row_start_h):min(dim, row_end_h), :] = 1.0
        # Vertical arm
        col_start_v = cy - arm_width // 2
        col_end_v = col_start_v + arm_width
        mat[:, max(0, col_start_v):min(dim, col_end_v)] = 1.0 # Overwrites center but that's fine
    else:
        raise ValueError(f"Unsupported shape_type: {shape_type}")

    return mat

def calculate_rotation_quality(original: NDArray, standard_rotated: NDArray, corrected_rotated: NDArray) -> Dict[str, float]:
    """
    Calculates metrics to evaluate the quality of the standard and corrected rotations.

    Args:
        original: The initial matrix shape.
        standard_rotated: The matrix after standard rotation.
        corrected_rotated: The matrix after corrected rotation.

    Returns:
        A dictionary containing:
        - 'overlap_standard': Sum of element-wise product of original and standard_rotated.
        - 'overlap_corrected': Sum of element-wise product of original and corrected_rotated.
        - 'difference_norm': Frobenius norm of the difference between corrected and standard rotations.
    """
    # Ensure inputs are valid
    if not (original.shape == standard_rotated.shape == corrected_rotated.shape):
        raise ValueError("All input matrices must have the same shape.")
        
    # Overlap often measures similarity (assuming non-negative values)
    overlap_std = float(np.sum(original * standard_rotated))
    overlap_corr = float(np.sum(original * corrected_rotated))

    # Norm of the difference measures how much the correction changed the result
    diff_norm = float(np.linalg.norm(corrected_rotated - standard_rotated, 'fro'))

    return {
        'overlap_standard': overlap_std,
        'overlap_corrected': overlap_corr,
        'difference_norm': diff_norm,
    }

##############################################
# Experiment and Visualization
##############################################

def display_matrices_graphical(original: NDArray, standard_rotated: NDArray, corrected_rotated: NDArray,
                               angle: float, quant_level: int,
                               strategy_info: str, modulus: int,
                               metrics: Dict[str, float]) -> Any: # Returns matplotlib Figure
    """
    Displays the original, standard rotated, and corrected rotated matrices side-by-side.

    Args:
        original: The initial matrix shape.
        standard_rotated: Matrix after standard rotation.
        corrected_rotated: Matrix after corrected rotation.
        angle: Rotation angle in degrees.
        quant_level: Quantization level used.
        strategy_info: String describing the correction strategy used (e.g., "LemmaBased(0.005)").
        modulus: Modulus of the phase space.
        metrics: Dictionary of calculated quality metrics.

    Returns:
        The matplotlib Figure object.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6)) # Increased size for 3 plots

    # Original Matrix
    im0 = axes[0].imshow(original, cmap='gray', interpolation='nearest', vmin=0, vmax=max(original.max(), 1e-6))
    axes[0].set_title("Original Shape")
    axes[0].axis('off')
    #fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04) # Optional colorbar

    # Standard Rotation
    im1 = axes[1].imshow(standard_rotated, cmap='gray', interpolation='nearest', vmin=0, vmax=max(standard_rotated.max(), 1e-6))
    axes[1].set_title(f"Standard Rotation\nOverlap w/ Original: {metrics['overlap_standard']:.2f}")
    axes[1].axis('off')
    #fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04) # Optional colorbar

    # Corrected Rotation
    im2 = axes[2].imshow(corrected_rotated, cmap='gray', interpolation='nearest', vmin=0, vmax=max(corrected_rotated.max(), 1e-6))
    axes[2].set_title(
        f"Corrected Rotation ({strategy_info})\n"
        f"Overlap w/ Original: {metrics['overlap_corrected']:.2f}\n"
        f"Diff Norm from Std: {metrics['difference_norm']:.2f}"
    )
    axes[2].axis('off')
    #fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04) # Optional colorbar


    fig.suptitle(
        f"Rotation Comparison\nAngle: {angle:.1f}°, Quantization Level: {quant_level}, Modulus: {modulus}"
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.94]) # Adjust layout to prevent title overlap
    return fig

def run_experiment_parameter_variation(
    dim: int,
    modulus: int,
    fixed_quantization_level: int,
    fixed_rotation_angle_deg: float,
    correction_strategy: CorrectionStrategy,
    strategy_name: str, # Pass name for storage
    param_name: str, # Name of the parameter being varied
    param_values: List[Union[int, float]],
    shape_type: str
) -> List[Dict[str, Any]]: # Changed return type hint for flexibility
    """
    Runs rotation experiments while varying one parameter.

    Args:
        dim: Dimension of the matrix/phase space.
        modulus: Modulus of the phase space.
        fixed_quantization_level: The quantization level used when not varying it.
        fixed_rotation_angle_deg: The rotation angle used when not varying it.
        correction_strategy: The CorrectionStrategy instance to use.
        strategy_name: A string identifier for the strategy.
        param_name: The name of the parameter to vary ('quantization_level' or 'rotation_angle_deg').
        param_values: A list of values for the parameter being varied.
        shape_type: The type of initial shape matrix to use.

    Returns:
        A list of dictionaries, where each dictionary contains the metrics
        and the parameter value for one run.
    """
    results = []
    phase_space = ModularPhaseSpace(dim, modulus)
    initial_matrix = create_shape_matrix(shape_type, dim)

    for val in param_values:
        # Determine current parameters for this run
        current_quant_level = int(val) if param_name == 'quantization_level' else fixed_quantization_level
        current_angle_deg = float(val) if param_name == 'rotation_angle_deg' else fixed_rotation_angle_deg

        # Setup operators and rotation matrix for current parameters
        weyl_x = ModularWeylXOperator(phase_space, current_quant_level)
        weyl_p = ModularWeylPOperator(phase_space, current_quant_level)
        rotation_matrix = discrete_rotation_operator(current_angle_deg, dim)

        # Perform standard rotation
        standard_rotated = apply_discrete_rotation(initial_matrix, rotation_matrix)

        # Get correction matrix and perform corrected rotation
        correction_matrix = correction_strategy.get_correction_matrix(initial_matrix, current_angle_deg, weyl_x, weyl_p)
        corrected_rotated = apply_modular_corrected_rotation(initial_matrix, rotation_matrix, correction_matrix)

        # Calculate quality metrics (comparing standard and corrected to original)
        metrics = calculate_rotation_quality(initial_matrix, standard_rotated, corrected_rotated)

        # Add metadata to the results
        metrics[param_name] = val # Store the value of the parameter varied
        metrics['quantization_level'] = current_quant_level # Store actual level used
        metrics['rotation_angle_deg'] = current_angle_deg # Store actual angle used
        metrics['strategy'] = strategy_name # Store strategy name
        metrics['modulus'] = modulus
        metrics['dimension'] = dim

        results.append(metrics)

    return results

def plot_experiment_results(
    results: List[Dict[str, Any]],
    param_name: str,
    modulus: int,
    strategy_name: str,
    shape_type: str
) -> None:
    """
    Plots the results from a parameter variation experiment.

    Args:
        results: The list of dictionaries returned by run_experiment_parameter_variation.
        param_name: The name of the parameter that was varied.
        modulus: The modulus used in the experiment.
        strategy_name: The name of the correction strategy used.
        shape_type: The shape used in the experiment.
    """
    if not results:
        print("No results to plot.")
        return

    # Extract data for plotting
    param_vals = [r[param_name] for r in results]
    overlap_std = [r['overlap_standard'] for r in results]
    overlap_corr = [r['overlap_corrected'] for r in results]
    diff_norm = [r['difference_norm'] for r in results]

    plt.figure(figsize=(10, 6))

    plt.plot(param_vals, overlap_std, 'o-', label='Standard Rotation Overlap w/ Original')
    plt.plot(param_vals, overlap_corr, 's-', label=f'Corrected ({strategy_name}) Overlap w/ Original')
    # Plot difference norm on a secondary y-axis if scales differ significantly
    # plt.plot(param_vals, diff_norm, '^-', label='Norm(Corrected - Standard)') # Alternative plotting

    # Use secondary y-axis for difference norm for better visualization if scales differ
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    line3, = ax2.plot(param_vals, diff_norm, '^--', color='green', label='Norm(Corrected - Standard)')
    ax2.set_ylabel('Difference Norm', color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines = lines1 + [line3]
    labels = labels1 + [line3.get_label()]
    ax1.legend(lines, labels, loc='best')


    ax1.set_xlabel(f'{param_name.replace("_", " ").title()}')
    ax1.set_ylabel('Overlap Metric')
    plt.title(f'Rotation Quality vs {param_name.replace("_", " ").title()}\n'
              f'(Shape: {shape_type}, Modulus: {modulus}, Strategy: {strategy_name})')
    ax1.grid(True, axis='y', linestyle=':')
    plt.show()


# Guard execution for when the script is run directly
if __name__ == "__main__":

    # --- Experiment Parameters ---
    DIM = 64          # Dimension of the matrix (e.g., 32, 50, 64)
    MODULUS = 64      # Modulus for phase space (often same as DIM)
    QUANT_LEVEL = 3   # Quantization level (shift amount)
    ANGLE_DEG = 45.0  # Rotation angle in degrees
    SHAPE = 'circle' # 'square', 'circle', 'triangle', 'line', 'cross'

    # Correction strategy parameters
    LEMMA_CORR_FACTOR = 0.01
    NOISE_CORR_FACTOR = 0.05

    # Choose strategy
    # strategy_instance = LemmaBasedCorrectionStrategy(correction_factor=LEMMA_CORR_FACTOR)
    # strategy_info_str = f"LemmaBased({LEMMA_CORR_FACTOR})" # For display
    # strategy_name_str = "Lemma"                          # For storing results
    
    strategy_instance = RandomNoiseCorrectionStrategy(noise_factor=NOISE_CORR_FACTOR)
    strategy_info_str = f"RandomNoise({NOISE_CORR_FACTOR})" # For display
    strategy_name_str = "Noise"                           # For storing results

    # --- Single Run Demonstration ---
    print("--- Running Single Demonstration ---")
    phase_space = ModularPhaseSpace(DIM, MODULUS)
    weyl_x = ModularWeylXOperator(phase_space, QUANT_LEVEL)
    weyl_p = ModularWeylPOperator(phase_space, QUANT_LEVEL)
    rotation_matrix = discrete_rotation_operator(ANGLE_DEG, DIM)
    initial_matrix = create_shape_matrix(SHAPE, DIM)

    # Apply standard rotation
    standard_rotated_matrix = apply_discrete_rotation(initial_matrix, rotation_matrix)

    # Apply corrected rotation
    correction_matrix = strategy_instance.get_correction_matrix(initial_matrix, ANGLE_DEG, weyl_x, weyl_p)
    corrected_rotated_matrix = apply_modular_corrected_rotation(initial_matrix, rotation_matrix, correction_matrix)

    # Calculate metrics
    quality_metrics = calculate_rotation_quality(initial_matrix, standard_rotated_matrix, corrected_rotated_matrix)

    # Display results
    fig_single = display_matrices_graphical(
        initial_matrix, standard_rotated_matrix, corrected_rotated_matrix,
        ANGLE_DEG, QUANT_LEVEL, strategy_info_str, MODULUS, quality_metrics
    )
    plt.show()

    # --- Parameter Variation Example: Varying Angle ---
    print("\n--- Running Parameter Variation (Angle) ---")
    angle_values = np.linspace(0, 90, 10) # Vary angle from 0 to 90 degrees
    results_angle_variation = run_experiment_parameter_variation(
        dim=DIM,
        modulus=MODULUS,
        fixed_quantization_level=QUANT_LEVEL,
        fixed_rotation_angle_deg=ANGLE_DEG, # This will be overridden by param_values
        correction_strategy=strategy_instance,
        strategy_name=strategy_name_str,
        param_name='rotation_angle_deg',
        param_values=list(angle_values),
        shape_type=SHAPE
    )
    # Plot angle variation results
    plot_experiment_results(results_angle_variation, 'rotation_angle_deg', MODULUS, strategy_name_str, SHAPE)


    # --- Parameter Variation Example: Varying Quantization Level ---
    print("\n--- Running Parameter Variation (Quantization Level) ---")
    quantization_values = list(range(1, MODULUS // 4)) # Vary quantization level up to Modulus/4
    results_quant_variation = run_experiment_parameter_variation(
        dim=DIM,
        modulus=MODULUS,
        fixed_quantization_level=QUANT_LEVEL, # This will be overridden by param_values
        fixed_rotation_angle_deg=ANGLE_DEG,
        correction_strategy=strategy_instance,
        strategy_name=strategy_name_str,
        param_name='quantization_level',
        param_values=quantization_values,
        shape_type=SHAPE
    )
    # Plot quantization level variation results
    plot_experiment_results(results_quant_variation, 'quantization_level', MODULUS, strategy_name_str, SHAPE)

    print("\n--- Experiment Finished ---")
