import numpy as np

def weyl_x_operator(matrix_dim, quantization_level=1):
    """Weyl X operator (same as before)."""
    def operator_func(matrix):
        if not isinstance(matrix, np.ndarray) or matrix.shape != (matrix_dim, matrix_dim):
            raise ValueError("Input must be a square NumPy matrix of specified dimension.")
        shifted_matrix = np.roll(matrix, shift=quantization_level, axis=0)
        return shifted_matrix
    return operator_func

def weyl_p_operator(matrix_dim, quantization_level=1):
    """Weyl P operator (same as before)."""
    def operator_func(matrix):
        if not isinstance(matrix, np.ndarray) or matrix.shape != (matrix_dim, matrix_dim):
            raise ValueError("Input must be a square NumPy matrix of specified dimension.")
        shifted_matrix = np.roll(matrix, shift=quantization_level, axis=1)
        return shifted_matrix
    return operator_func

def rotation_operator(angle_degrees):
    """Rotation operator (same as before)."""
    angle_radians = np.radians(angle_degrees)
    cos_theta = np.cos(angle_radians)
    sin_theta = np.sin(angle_radians)
    rotation_mat = np.array([[cos_theta, -sin_theta],
                             [sin_theta,  cos_theta]])
    return rotation_mat

def apply_rotation(matrix, rotation_mat):
    """Apply rotation (same as before)."""
    if matrix.shape != (2, 2) or rotation_mat.shape != (2, 2):
        raise ValueError("Matrices must be 2x2 for simplified rotation example.")
    rotated_matrix = rotation_mat @ matrix @ rotation_mat.T
    return rotated_matrix

def estimate_quantization_effect(matrix, weyl_x, weyl_p):
    """Estimate quantization effect (same as before)."""
    matrix_xp = weyl_p(weyl_x(matrix.copy()))
    matrix_px = weyl_x(weyl_p(matrix.copy()))
    difference_matrix = matrix_xp - matrix_px
    quantization_estimate = np.linalg.norm(difference_matrix, 'fro')
    return quantization_estimate

def build_irrational_correction_matrix(initial_matrix, rotation_angle_degrees, weyl_x, weyl_p, correction_factor=0.005):
    """
    Builds an "irrational factor correction matrix" based on a simplified "lemma system"
    related to Weyl algebra and quantization effects.

    Lemma System (Simplified Guiding Principles):

    Lemma 1 (Quantization Imprecision Indicator): The magnitude of non-commutativity between
    discretized Weyl X and P operators, as measured by the Frobenius norm of their commutator
    applied to the initial matrix, serves as an indicator of the level of quantization-induced
    imprecision in the system for the given matrix.

    Lemma 2 (Correction Proportionality): The correction applied to mitigate irrational factor
    inaccuracies during rotation should be proportional to the estimated quantization imprecision.
    A higher degree of imprecision suggests a need for a more significant correction.

    Lemma 3 (Correction acts on trace): Due to the nature of quantization errors potentially accumulating
    in matrix operations involving irrational numbers during rotation, the correction should primarily
    affect the overall scale or trace-like properties of the matrix rather than directional components
    in this simplified model. Scaling the identity matrix serves as a rudimentary way to adjust the 'magnitude'
    represented by the matrix.

    Implementation based on these simplified "lemmas":

    1. Estimate Quantization Imprecision (Lemma 1): Use the `estimate_quantization_effect` function
       to get a scalar value representing the level of imprecision for the given initial matrix and Weyl operators.

    2. Determine Correction Scale (Lemma 2 & 3): Scale the correction based on the quantization estimate
       and a `correction_factor`. The higher the quantization estimate, the larger the correction scale.
       We still use scaling of the identity matrix as a rudimentary correction affecting the overall magnitude
       (as hinted by Lemma 3).

    3. Build Correction Matrix: Create a correction matrix (diagonal for simplicity, scaling the identity)
       that is intended to pre-adjust the matrix *before* rotation, aiming to compensate for irrationality
       and quantization.

    Args:
        initial_matrix (np.ndarray): The initial matrix being rotated. The correction is context-dependent.
        rotation_angle_degrees (float): The rotation angle.  While not directly used in the correction matrix itself
                                       in this version, it could be incorporated to further refine based on angle.
        weyl_x (function): Weyl X operator function.
        weyl_p (function): Weyl P operator function.
        correction_factor (float):  Scaling factor to control correction strength (tuned empirically).

    Returns:
        np.ndarray: A 2x2 correction matrix derived from the "lemma system".
    """
    quantization_estimate = estimate_quantization_effect(initial_matrix, weyl_x, weyl_p)

    # Correction scale directly proportional to quantization estimate (Lemma 2 & 3)
    correction_scale = correction_factor * quantization_estimate

    # Correction matrix: Scaled identity (Lemma 3 - rudimentary magnitude adjustment)
    correction_matrix = np.eye(2) * (1.0 - correction_scale)

    return correction_matrix


def apply_corrected_rotation(matrix, rotation_mat, correction_matrix):
    """Apply corrected rotation (same as before)."""
    if matrix.shape != (2, 2) or rotation_mat.shape != (2, 2) or correction_matrix.shape != (2,2):
        raise ValueError("Matrices must be 2x2 for corrected rotation example.")

    corrected_matrix_pre_rotation = correction_matrix @ matrix
    rotated_matrix_corrected = rotation_mat @ corrected_matrix_pre_rotation @ rotation_mat.T
    return rotated_matrix_corrected


def demonstrate_weyl_algebra_and_rotation(matrix_dim, quantization_level, rotation_angle_deg):
    """Demonstrates with improved correction matrix using lemma system."""
    print(f"\n--- Demonstrating Weyl Algebra, Rotation, and Lemma-Based Correction (Quantization Level: {quantization_level}) ---")

    initial_matrix = np.eye(matrix_dim)

    print("\nInitial Matrix:")
    print(initial_matrix)

    weyl_x = weyl_x_operator(matrix_dim, quantization_level)
    weyl_p = weyl_p_operator(matrix_dim, quantization_level)

    if matrix_dim == 2:
        rot_mat = rotation_operator(rotation_angle_deg)
        print(f"\nRotation Matrix ({rotation_angle_deg} degrees):")
        print(rot_mat)

        # Estimate Quantization Effect (as before)
        quantization_effect = estimate_quantization_effect(initial_matrix.copy(), weyl_x, weyl_p)
        print(f"\nEstimated Quantization Effect (Non-commutativity Norm): {quantization_effect:.6f}")

        # Build Correction Matrix using Lemma System - Now core part of demonstration
        correction_mat = build_irrational_correction_matrix(initial_matrix.copy(), rotation_angle_deg, weyl_x, weyl_p) # Pass Weyl operators and initial matrix
        print("\nIrrational Factor Correction Matrix (Lemma-Based):")
        print(correction_mat)

        # Apply Corrected Rotation (as before)
        rotated_matrix_standard = apply_rotation(initial_matrix.copy(), rot_mat)
        rotated_matrix_corrected = apply_corrected_rotation(initial_matrix.copy(), rot_mat, correction_mat)

        print("\nStandard Rotated Matrix:")
        print(rotated_matrix_standard)
        print("\nCorrected Rotated Matrix (Lemma-Based Correction):")
        print(rotated_matrix_corrected)

        difference_corrected_standard = rotated_matrix_corrected - rotated_matrix_standard
        print("\nDifference between Corrected and Standard Rotation:")
        print(difference_corrected_standard)
        diff_norm = np.linalg.norm(difference_corrected_standard, 'fro')
        print(f"\nFrobenius Norm of Difference: {diff_norm:.6f}")
        if diff_norm > 1e-9:
            print("\nLemma-based correction matrix does alter the rotation result.")
        else:
            print("\nLemma-based correction matrix has negligible effect (or correction factor is too small).")

    print("\n--- End of Demonstration ---")


if __name__ == "__main__":
    matrix_dimension_2d = 2
    quantization = 1
    rotation_angle = 45

    demonstrate_weyl_algebra_and_rotation(matrix_dimension_2d, quantization, rotation_angle)
    demonstrate_weyl_algebra_and_rotation(matrix_dimension_2d, quantization=2, rotation_angle=45) # Higher quantization
    demonstrate_weyl_algebra_and_rotation(matrix_dimension_2d, quantization=1, rotation_angle=30) # Different angle
    demonstrate_weyl_algebra_and_rotation(matrix_dimension_2d, quantization=1, rotation_angle=75) # Another angle
