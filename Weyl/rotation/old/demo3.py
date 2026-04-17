import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Re-use ModularPhaseSpace, Operator, ModularWeylXOperator, ModularWeylPOperator,
# discrete_rotation_operator, apply_discrete_rotation,
# estimate_quantization_effect_modular, LemmaBasedCorrectionStrategy,
# apply_modular_corrected_rotation classes and functions from the previous response.
# (Code for these classes and functions is assumed to be copy-pasted here for completeness)

# ---  Paste ModularPhaseSpace, Operator, ModularWeylXOperator, ModularWeylPOperator,
# ---  discrete_rotation_operator, apply_discrete_rotation,
# ---  estimate_quantization_effect_modular, LemmaBasedCorrectionStrategy,
# ---  apply_modular_corrected_rotation code from previous response HERE ---
# Placeholder - In real code, paste the class and function definitions from the previous improved response.

# --- BEGIN -  Code from previous response (ModularPhaseSpace, Operator, etc.) ---
class ModularPhaseSpace:
    """... (ModularPhaseSpace class definition from previous response) ..."""
    def __init__(self, dimension, modulus):
        self.dimension = dimension
        self.modulus = modulus
        if modulus <= 0:
            raise ValueError("Modulus must be a positive integer.")
        self.indices = np.arange(dimension) % modulus

    def get_index(self, index):
        return index % self.modulus

    def shift_index(self, index, shift):
        return (index + shift) % self.modulus

    def get_modular_indices(self):
        return self.indices

    def __repr__(self):
        return f"ModularPhaseSpace(dimension={self.dimension}, modulus={self.modulus})"


class Operator:
    """... (Operator class definition from previous response) ..."""
    def __init__(self, phase_space):
        if not isinstance(phase_space, ModularPhaseSpace):
            raise TypeError("Operator must be initialized with a ModularPhaseSpace object.")
        self.phase_space = phase_space

    def operate(self, matrix):
        raise NotImplementedError("Subclasses must implement the operate method.")

    def __call__(self, matrix):
        return self.operate(matrix)


class ModularWeylXOperator(Operator):
    """... (ModularWeylXOperator class definition from previous response) ..."""
    def __init__(self, phase_space, quantization_level=1):
        super().__init__(phase_space)
        self.quantization_level = quantization_level

    def operate(self, matrix):
        if not isinstance(matrix, np.ndarray) or matrix.shape != (self.phase_space.dimension, self.phase_space.dimension):
            raise ValueError("Input must be a square NumPy matrix of specified dimension.")

        shifted_matrix = np.zeros_like(matrix)
        for i in range(self.phase_space.dimension):
            shifted_row_index = self.phase_space.shift_index(i, self.quantization_level)
            shifted_matrix[shifted_row_index, :] = matrix[i, :]
        return shifted_matrix


class ModularWeylPOperator(Operator):
    """... (ModularWeylPOperator class definition from previous response) ..."""
    def __init__(self, phase_space, quantization_level=1):
        super().__init__(phase_space)
        self.quantization_level = quantization_level

    def operate(self, matrix):
        if not isinstance(matrix, np.ndarray) or matrix.shape != (self.phase_space.dimension, self.phase_space.dimension):
            raise ValueError("Input must be a square NumPy matrix of specified dimension.")

        shifted_matrix = np.zeros_like(matrix)
        for j in range(self.phase_space.dimension):
            shifted_col_index = self.phase_space.shift_index(j, self.quantization_level)
            shifted_matrix[:, shifted_col_index] = matrix[:, j]
        return shifted_matrix


def discrete_rotation_operator(angle_degrees, discrete_space_dimension):
    """... (discrete_rotation_operator function definition from previous response) ..."""
    angle_radians = np.radians(angle_degrees)
    continuous_rotation_matrix_2d = np.array([[np.cos(angle_radians), -np.sin(angle_radians)],
                                              [np.sin(angle_radians),  np.cos(angle_radians)]])

    discrete_rotation_matrix = np.eye(discrete_space_dimension, dtype=float)
    if discrete_space_dimension >= 2:
        discrete_rotation_matrix[0:2, 0:2] = continuous_rotation_matrix_2d

    return discrete_rotation_matrix


def apply_discrete_rotation(matrix, discrete_rotation_mat):
    """... (apply_discrete_rotation function definition from previous response) ..."""
    if matrix.shape != discrete_rotation_mat.shape:
        raise ValueError("Matrix and rotation matrix dimensions must match for discrete rotation.")
    rotated_matrix = discrete_rotation_mat @ matrix @ discrete_rotation_mat.T
    return rotated_matrix


def estimate_quantization_effect_modular(matrix, modular_weyl_x, modular_weyl_p):
    """... (estimate_quantization_effect_modular function definition from previous response) ..."""
    matrix_xp = modular_weyl_p(modular_weyl_x(matrix.copy()))
    matrix_px = modular_weyl_x(modular_weyl_p(matrix.copy()))
    difference_matrix = matrix_xp - matrix_px
    quantization_estimate = np.linalg.norm(difference_matrix, 'fro')
    return quantization_estimate


class CorrectionStrategy:
    """... (CorrectionStrategy class definition from previous response) ..."""
    def __init__(self):
        pass
    def get_correction_matrix(self, initial_matrix, rotation_angle_degrees, weyl_x, weyl_p):
        raise NotImplementedError("Subclasses must implement get_correction_matrix.")


class LemmaBasedCorrectionStrategy(CorrectionStrategy):
    """... (LemmaBasedCorrectionStrategy class definition from previous response) ..."""
    def __init__(self, correction_factor=0.005):
        super().__init__()
        self.correction_factor = correction_factor

    def get_correction_matrix(self, initial_matrix, rotation_angle_degrees, modular_weyl_x, modular_weyl_p):
        quantization_estimate = estimate_quantization_effect_modular(initial_matrix, modular_weyl_x, modular_weyl_p)
        correction_scale = self.correction_factor * quantization_estimate
        correction_matrix = np.eye(initial_matrix.shape[0]) * (1.0 - correction_scale)
        return correction_matrix


def apply_modular_corrected_rotation(matrix, discrete_rotation_mat, correction_matrix):
    """... (apply_modular_corrected_rotation function definition from previous response) ..."""
    if matrix.shape != discrete_rotation_mat.shape or matrix.shape != correction_matrix.shape:
        raise ValueError("Matrix and rotation/correction matrices must have compatible dimensions.")

    corrected_matrix_pre_rotation = correction_matrix @ matrix
    rotated_matrix_corrected = discrete_rotation_mat @ corrected_matrix_pre_rotation @ discrete_rotation_mat.T
    return rotated_matrix_corrected
# --- END - Code from previous response ---


def create_shape_matrix(shape_type, matrix_dim):
    """Creates a matrix representing a simple geometric shape (same as before)."""
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

    elif shape_type == 'triangle': # Rudimentary triangle approximation
        base_width = matrix_dim // 3
        height = matrix_dim // 2
        for i in range(matrix_dim):
            for j in range(matrix_dim):
                if i >= center_y and i <= center_y + height and abs(j - center_x) <= base_width * (1 - (i - center_y) / height) / 2:
                     matrix[i, j] = 1

    return matrix


def calculate_rotation_quality(original_shape, rotated_shape_standard, rotated_shape_corrected):
    """
    Calculates a simple "rotation quality" metric based on shape overlap and difference.

    This is a heuristic metric to quantify how "well" the shapes are rotated
    and if the corrected rotation appears "better" than the standard rotation.

    Metric components:
    1. Overlap with original shape (inverted standard and corrected): Lower overlap after rotation is generally better (less distortion).
    2. Frobenius norm of difference between standard and corrected rotated shapes: Smaller difference might be desirable
       if the correction is intended to be subtle and refine the rotation, but this is context-dependent.

    Args:
        original_shape (np.ndarray): Original shape matrix.
        rotated_shape_standard (np.ndarray): Standard rotated matrix.
        rotated_shape_corrected (np.ndarray): Corrected rotated matrix.

    Returns:
        dict: Dictionary containing quality metrics: 'overlap_standard', 'overlap_corrected', 'difference_norm'.
    """
    overlap_standard = np.sum(original_shape * rotated_shape_standard) # Element-wise overlap
    overlap_corrected = np.sum(original_shape * rotated_shape_corrected)

    difference_matrix = rotated_shape_corrected - rotated_shape_standard
    difference_norm = np.linalg.norm(difference_matrix, 'fro')

    return {
        'overlap_standard': overlap_standard,
        'overlap_corrected': overlap_corrected,
        'difference_norm': difference_norm,
    }


def display_matrices_graphical(matrix_standard, matrix_corrected, angle, quantization, correction_factor, modulus, quality_metrics=None):
    """
    Displays matrices with improved information and rotation quality metrics.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7)) # Slightly wider figure

    im1 = axes[0].imshow(matrix_standard, cmap='gray', interpolation='nearest')
    title_standard = 'Standard Discrete Rotation'
    if quality_metrics:
        title_standard += f"\nOverlap: {quality_metrics['overlap_standard']:.2f}" # Display overlap in title
    axes[0].set_title(title_standard)
    axes[0].axis('off')

    im2 = axes[1].imshow(matrix_corrected, cmap='gray', interpolation='nearest')
    title_corrected = 'Corrected Rotation (Lemma-Based)'
    if quality_metrics:
        title_corrected += f"\nOverlap: {quality_metrics['overlap_corrected']:.2f}, Diff Norm: {quality_metrics['difference_norm']:.2f}" # Display metrics
    axes[1].set_title(title_corrected)
    axes[1].axis('off')

    fig.suptitle(f'Rotation Angle: {angle}° | Quantization Level: {quantization} | Correction Factor: {correction_factor:.4f} | Modulus: {modulus}')

    plt.tight_layout(rect=[0, 0.03, 1, 0.93]) # Adjusted layout
    return fig, axes, [im1, im2]


def demonstrate_modular_weyl_algebra_and_rotation_graphical(matrix_dim, modulus, quantization_level_initial, rotation_angle_initial, correction_factor_initial, shape_type='square'):
    """Graphical demonstration with rotation quality metrics and display."""
    initial_shape_matrix = create_shape_matrix(shape_type, matrix_dim)
    phase_space = ModularPhaseSpace(matrix_dim, modulus)
    modular_weyl_x = ModularWeylXOperator(phase_space, quantization_level_initial)
    modular_weyl_p = ModularWeylPOperator(phase_space, quantization_level_initial)
    discrete_rot_mat = discrete_rotation_operator(rotation_angle_initial, matrix_dim)
    lemma_correction = LemmaBasedCorrectionStrategy(correction_factor=correction_factor_initial)

    rotated_matrix_standard_discrete = apply_discrete_rotation(initial_shape_matrix.copy(), discrete_rot_mat)
    correction_mat = lemma_correction.get_correction_matrix(initial_shape_matrix.copy(), rotation_angle_initial, modular_weyl_x, modular_weyl_p)
    rotated_matrix_corrected_modular = apply_modular_corrected_rotation(initial_shape_matrix.copy(), discrete_rot_mat, correction_mat)

    quality_metrics = calculate_rotation_quality(initial_shape_matrix, rotated_matrix_standard_discrete, rotated_matrix_corrected_modular) # Calculate metrics

    fig, axes, images = display_matrices_graphical(rotated_matrix_standard_discrete, rotated_matrix_corrected_modular,
                                        rotation_angle_initial, quantization_level_initial, correction_factor_initial, modulus, quality_metrics) # Pass metrics to display

    ax_angle = plt.axes([0.25, 0.01, 0.45, 0.02])
    angle_slider = Slider(ax=ax_angle, label='Rotation Angle', valmin=0, valmax=360, valinit=rotation_angle_initial, valstep=1)

    ax_quant = plt.axes([0.25, 0.04, 0.45, 0.02])
    quant_slider = Slider(ax=ax_quant, label='Quantization Level', valmin=1, valmax=5, valinit=quantization_level_initial, valstep=1, valfmt='%i')

    ax_corr = plt.axes([0.25, 0.07, 0.45, 0.02])
    corr_slider = Slider(ax=ax_corr, label='Correction Factor', valmin=0.0, valmax=0.1, valinit=correction_factor_initial, valstep=0.001)


    def update(val):
        angle = angle_slider.val
        quantization = int(quant_slider.val)
        correction_factor = corr_slider.val

        phase_space_update = ModularPhaseSpace(matrix_dim, modulus)
        modular_weyl_x_update = ModularWeylXOperator(phase_space_update, quantization)
        modular_weyl_p_update = ModularWeylPOperator(phase_space_update, quantization)
        discrete_rot_mat_update = discrete_rotation_operator(angle, matrix_dim)
        lemma_correction_update = LemmaBasedCorrectionStrategy(correction_factor=correction_factor)

        rotated_matrix_standard_discrete_update = apply_discrete_rotation(initial_shape_matrix.copy(), discrete_rot_mat_update)
        correction_mat_update = lemma_correction_update.get_correction_matrix(initial_shape_matrix.copy(), angle, modular_weyl_x_update, modular_weyl_p_update)
        rotated_matrix_corrected_modular_update = apply_modular_corrected_rotation(initial_shape_matrix.copy(), discrete_rot_mat_update, correction_mat_update)

        quality_metrics_update = calculate_rotation_quality(initial_shape_matrix, rotated_matrix_standard_discrete_update, rotated_matrix_corrected_modular_update) # Recalculate metrics

        images[0].set_array(rotated_matrix_standard_discrete_update)
        images[1].set_array(rotated_matrix_corrected_modular_update)
        axes[0].set_title(f'Standard Discrete Rotation\nOverlap: {quality_metrics_update["overlap_standard"]:.2f}') # Update titles with metrics
        axes[1].set_title(f'Corrected Rotation (Lemma-Based)\nOverlap: {quality_metrics_update["overlap_corrected"]:.2f}, Diff Norm: {quality_metrics_update["difference_norm"]:.2f}')
        fig.suptitle(f'Rotation Angle: {angle}° | Quantization Level: {quantization} | Correction Factor: {correction_factor:.4f} | Modulus: {modulus}')
        fig.canvas.draw_idle()

    angle_slider.on_changed(update)
    quant_slider.on_changed(update)
    corr_slider.on_changed(update)

    plt.show()


def run_quantization_experiment(matrix_dim, modulus, rotation_angle_deg, correction_strategy, quantization_levels):
    """
    Runs a structured experiment varying quantization levels and measures rotation quality.

    Args:
        matrix_dim (int): Matrix dimension.
        modulus (int): Modulus value.
        rotation_angle_deg (float): Rotation angle (fixed for this experiment).
        correction_strategy (CorrectionStrategy): Correction strategy to use (or None for no correction).
        quantization_levels (list): List of quantization levels to test.

    Returns:
        dict: Dictionary containing quality metrics for standard and corrected rotations
              across different quantization levels.
    """
    initial_shape_matrix = create_shape_matrix('square', matrix_dim) # Fixed shape for experiment
    phase_space = ModularPhaseSpace(matrix_dim, modulus)
    discrete_rot_mat = discrete_rotation_operator(rotation_angle_deg, matrix_dim)
    modular_weyl_x_base = ModularWeylXOperator(phase_space) # Base operators (quantization level will vary)
    modular_weyl_p_base = ModularWeylPOperator(phase_space)

    standard_overlaps = []
    corrected_overlaps = []
    difference_norms = []

    for quantization_level in quantization_levels:
        modular_weyl_x = ModularWeylXOperator(phase_space, quantization_level) # Create operators with current quantization
        modular_weyl_p = ModularWeylPOperator(phase_space, quantization_level)

        rotated_matrix_standard_discrete = apply_discrete_rotation(initial_shape_matrix.copy(), discrete_rot_mat)
        correction_mat = np.eye(matrix_dim) # Default no correction
        if correction_strategy:
            correction_mat = correction_strategy.get_correction_matrix(initial_shape_matrix.copy(), rotation_angle_deg, modular_weyl_x, modular_weyl_p)
        rotated_matrix_corrected_modular = apply_modular_corrected_rotation(initial_shape_matrix.copy(), discrete_rot_mat, correction_mat)

        quality_metrics = calculate_rotation_quality(initial_shape_matrix, rotated_matrix_standard_discrete, rotated_matrix_corrected_modular)
        standard_overlaps.append(quality_metrics['overlap_standard'])
        corrected_overlaps.append(quality_metrics['overlap_corrected'])
        difference_norms.append(quality_metrics['difference_norm'])

    return {
        'quantization_levels': quantization_levels,
        'standard_overlaps': standard_overlaps,
        'corrected_overlaps': corrected_overlaps,
        'difference_norms': difference_norms,
    }


def plot_experiment_results(experiment_results, rotation_angle_deg, modulus):
    """Plots the results of the quantization level experiment."""
    quant_levels = experiment_results['quantization_levels']
    standard_overlaps = experiment_results['standard_overlaps']
    corrected_overlaps = experiment_results['corrected_overlaps']
    difference_norms = experiment_results['difference_norms']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6)) # Two subplots for metrics

    axes[0].plot(quant_levels, standard_overlaps, marker='o', label='Standard Rotation')
    axes[0].plot(quant_levels, corrected_overlaps, marker='x', label='Corrected Rotation')
    axes[0].set_xlabel('Quantization Level')
    axes[0].set_ylabel('Shape Overlap with Original')
    axes[0].set_title(f'Shape Overlap vs. Quantization (Angle: {rotation_angle_deg}°, Modulus: {modulus})')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(quant_levels, difference_norms, marker='s', linestyle='--', color='purple', label='Difference Norm (Corrected - Standard)')
    axes[1].set_xlabel('Quantization Level')
    axes[1].set_ylabel('Frobenius Norm of Difference')
    axes[1].set_title(f'Difference Norm vs. Quantization (Angle: {rotation_angle_deg}°, Modulus: {modulus})')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    matrix_dimension_graphical = 64
    modulus_value_graphical = 64
    quantization_initial = 1
    rotation_angle_initial = 30
    correction_factor_initial = 0.01
    shape = 'square'

    lemma_correction = LemmaBasedCorrectionStrategy(correction_factor=0.005)

    demonstrate_modular_weyl_algebra_and_rotation_graphical(matrix_dimension_graphical, modulus_value_graphical,
                                                            quantization_initial, rotation_angle_initial,
                                                            correction_factor_initial, shape_type=shape)

    # Run Experiment and Plot Results
    experiment_quantization_levels = [1, 2, 3, 4, 5]
    experiment_results = run_quantization_experiment(matrix_dimension_graphical, modulus_value_graphical,
                                                    rotation_angle_initial, lemma_correction, experiment_quantization_levels)
    plot_experiment_results(experiment_results, rotation_angle_initial, modulus_value_graphical)
