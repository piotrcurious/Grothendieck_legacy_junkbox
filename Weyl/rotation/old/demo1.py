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


def create_shape_matrix(shape_type, matrix_dim):
    """
    Creates a matrix representing a simple geometric shape.

    Shapes: 'square', 'circle', 'triangle' (rudimentary triangle approximation).

    Args:
        shape_type (str): Type of shape ('square', 'circle', 'triangle').
        matrix_dim (int): Dimension of the square matrix.

    Returns:
        np.ndarray: Matrix representing the shape (1s for shape, 0s for background).
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

    elif shape_type == 'triangle': # Rudimentary triangle approximation
        base_width = matrix_dim // 3
        height = matrix_dim // 2
        for i in range(matrix_dim):
            for j in range(matrix_dim):
                if i >= center_y and i <= center_y + height and abs(j - center_x) <= base_width * (1 - (i - center_y) / height) / 2:
                     matrix[i, j] = 1

    return matrix


def display_matrices(matrix_standard, matrix_corrected, angle, quantization, correction_factor, modulus):
    """
    Displays two matrices side-by-side using matplotlib, with parameter labels.

    Args:
        matrix_standard (np.ndarray): Standard rotated matrix.
        matrix_corrected (np.ndarray): Corrected rotated matrix.
        angle (float): Rotation angle.
        quantization (int): Quantization level.
        correction_factor (float): Correction factor.
        modulus (int): Modulus value.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6)) # Create figure and axes

    im1 = axes[0].imshow(matrix_standard, cmap='gray', interpolation='nearest') # Display matrices as grayscale images
    axes[0].set_title('Standard Discrete Rotation')
    axes[0].axis('off') # Hide axes ticks and labels

    im2 = axes[1].imshow(matrix_corrected, cmap='gray', interpolation='nearest')
    axes[1].set_title('Corrected Rotation (Lemma-Based)')
    axes[1].axis('off')

    fig.suptitle(f'Rotation Angle: {angle}° | Quantization Level: {quantization} | Correction Factor: {correction_factor:.4f} | Modulus: {modulus}') # Overall title

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to fit title
    return fig, axes, [im1, im2] # Return figure, axes, image objects for updating


def demonstrate_modular_weyl_algebra_and_rotation_graphical(matrix_dim, modulus, quantization_level_initial, rotation_angle_initial, correction_factor_initial, shape_type='square'):
    """
    Graphical demonstration of modular Weyl algebra, discrete rotation, and correction with sliders.
    """
    initial_shape_matrix = create_shape_matrix(shape_type, matrix_dim) # Create shape matrix
    phase_space = ModularPhaseSpace(matrix_dim, modulus)
    modular_weyl_x = ModularWeylXOperator(phase_space, quantization_level_initial)
    modular_weyl_p = ModularWeylPOperator(phase_space, quantization_level_initial)
    discrete_rot_mat = discrete_rotation_operator(rotation_angle_initial, matrix_dim)
    lemma_correction = LemmaBasedCorrectionStrategy(correction_factor=correction_factor_initial)

    rotated_matrix_standard_discrete = apply_discrete_rotation(initial_shape_matrix.copy(), discrete_rot_mat)
    correction_mat = lemma_correction.get_correction_matrix(initial_shape_matrix.copy(), rotation_angle_initial, modular_weyl_x, modular_weyl_p)
    rotated_matrix_corrected_modular = apply_modular_corrected_rotation(initial_shape_matrix.copy(), discrete_rot_mat, correction_mat)


    fig, axes, images = display_matrices(rotated_matrix_standard_discrete, rotated_matrix_corrected_modular,
                                        rotation_angle_initial, quantization_level_initial, correction_factor_initial, modulus)

    # --- Slider Creation ---
    ax_angle = plt.axes([0.25, 0.01, 0.45, 0.02]) # Position of angle slider
    angle_slider = Slider(ax=ax_angle, label='Rotation Angle', valmin=0, valmax=360, valinit=rotation_angle_initial, valstep=1)

    ax_quant = plt.axes([0.25, 0.04, 0.45, 0.02]) # Position of quantization slider
    quant_slider = Slider(ax=ax_quant, label='Quantization Level', valmin=1, valmax=5, valinit=quantization_level_initial, valstep=1, valfmt='%i') # Integer slider

    ax_corr = plt.axes([0.25, 0.07, 0.45, 0.02]) # Position of correction factor slider
    corr_slider = Slider(ax=ax_corr, label='Correction Factor', valmin=0.0, valmax=0.1, valinit=correction_factor_initial, valstep=0.001) # Finer step for correction


    def update(val):
        angle = angle_slider.val
        quantization = int(quant_slider.val) # Ensure quantization level is integer
        correction_factor = corr_slider.val

        phase_space_update = ModularPhaseSpace(matrix_dim, modulus) # Re-create phase space and operators with new params
        modular_weyl_x_update = ModularWeylXOperator(phase_space_update, quantization)
        modular_weyl_p_update = ModularWeylPOperator(phase_space_update, quantization)
        discrete_rot_mat_update = discrete_rotation_operator(angle, matrix_dim)
        lemma_correction_update = LemmaBasedCorrectionStrategy(correction_factor=correction_factor)

        rotated_matrix_standard_discrete_update = apply_discrete_rotation(initial_shape_matrix.copy(), discrete_rot_mat_update)
        correction_mat_update = lemma_correction_update.get_correction_matrix(initial_shape_matrix.copy(), angle, modular_weyl_x_update, modular_weyl_p_update)
        rotated_matrix_corrected_modular_update = apply_modular_corrected_rotation(initial_shape_matrix.copy(), discrete_rot_mat_update, correction_mat_update)

        images[0].set_array(rotated_matrix_standard_discrete_update) # Update image data
        images[1].set_array(rotated_matrix_corrected_modular_update)
        fig.suptitle(f'Rotation Angle: {angle}° | Quantization Level: {quantization} | Correction Factor: {correction_factor:.4f} | Modulus: {modulus}') # Update title
        fig.canvas.draw_idle() # Request re-draw


    angle_slider.on_changed(update)
    quant_slider.on_changed(update)
    corr_slider.on_changed(update)

    plt.show()


if __name__ == "__main__":
    matrix_dimension_graphical = 64 # Higher dimension for better shape visualization
    modulus_value_graphical = 64     # Match modulus to dimension for visual wrap-around effect
    quantization_initial = 1
    rotation_angle_initial = 30
    correction_factor_initial = 0.01
    shape = 'square' # Try 'circle', 'triangle', 'square'

    demonstrate_modular_weyl_algebra_and_rotation_graphical(matrix_dimension_graphical, modulus_value_graphical,
                                                            quantization_initial, rotation_angle_initial,
                                                            correction_factor_initial, shape_type=shape)
