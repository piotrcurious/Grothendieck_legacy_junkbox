import numpy as np

def create_modular_space(dimension, modulus):
    """
    Creates a modular space (integers modulo 'modulus') for matrix indices.

    Args:
        dimension (int): Dimension of the matrix (and phase space).
        modulus (int): Modulus for the finite field/ring. Should ideally be a prime number
                       for finite field properties, but can be integer for ring demo.

    Returns:
        np.ndarray: 1D array representing the modular space indices [0, 1, ..., dimension-1] mod modulus.
    """
    return np.arange(dimension) % modulus

def modular_weyl_x_operator(matrix_dim, modulus, quantization_level=1):
    """
    Modular Weyl X operator acting in a discretized phase space (integers mod 'modulus').

    This operator performs a cyclic shift of rows in a matrix, with indices treated
    as elements of Z_modulus (integers modulo modulus).

    Args:
        matrix_dim (int): Dimension of the matrix.
        modulus (int): Modulus of the finite field/ring.
        quantization_level (int): Quantization step (integer).

    Returns:
        function: Operator function for modular Weyl X.
    """
    def operator_func(matrix):
        if not isinstance(matrix, np.ndarray) or matrix.shape != (matrix_dim, matrix_dim):
            raise ValueError("Input must be a square NumPy matrix of specified dimension.")

        # Modular shift of rows
        shifted_matrix = np.zeros_like(matrix)
        for i in range(matrix_dim):
            shifted_row_index = (i + quantization_level) % modulus  # Modular arithmetic for index
            shifted_matrix[shifted_row_index, :] = matrix[i, :]
        return shifted_matrix
    return operator_func

def modular_weyl_p_operator(matrix_dim, modulus, quantization_level=1):
    """
    Modular Weyl P operator in discretized phase space (integers mod 'modulus').

    Performs cyclic shift of columns with modular index arithmetic.

    Args:
        matrix_dim (int): Dimension of matrix.
        modulus (int): Modulus of the finite field/ring.
        quantization_level (int): Quantization step.

    Returns:
        function: Operator function for modular Weyl P.
    """
    def operator_func(matrix):
        if not isinstance(matrix, np.ndarray) or matrix.shape != (matrix_dim, matrix_dim):
            raise ValueError("Input must be a square NumPy matrix of specified dimension.")

        # Modular shift of columns
        shifted_matrix = np.zeros_like(matrix)
        for j in range(matrix_dim):
            shifted_col_index = (j + quantization_level) % modulus # Modular arithmetic for index
            shifted_matrix[:, shifted_col_index] = matrix[:, j]
        return shifted_matrix
    return operator_func

def discrete_rotation_operator(angle_degrees, discrete_space_dimension):
    """
    Approximation of rotation in a discrete space.

    This is a simplified attempt to represent rotation in a discrete setting.
    True rotation in a discrete phase space is mathematically complex.
    This function generates a rotation matrix in continuous space and then
    discretizes it (by rounding) to create a discrete operator of size
    discrete_space_dimension x discrete_space_dimension.

    This is a very rough approximation and may not perfectly represent
    rotations in a formal discrete phase space.  More rigorous approaches
    would require representation theory on discrete groups or finite fields.

    Args:
        angle_degrees (float): Rotation angle.
        discrete_space_dimension (int): Dimension of the discrete space (and resulting matrix).

    Returns:
        np.ndarray: Discrete rotation matrix (approximation).
    """
    angle_radians = np.radians(angle_degrees)
    continuous_rotation_matrix_2d = np.array([[np.cos(angle_radians), -np.sin(angle_radians)],
                                              [np.sin(angle_radians),  np.cos(angle_radians)]])

    # Expand to discrete space dimension (identity outside 2x2 rotation subspace - very simplified)
    discrete_rotation_matrix = np.eye(discrete_space_dimension, dtype=float) # Initialize as identity
    if discrete_space_dimension >= 2:
        discrete_rotation_matrix[0:2, 0:2] = continuous_rotation_matrix_2d # Embed 2D rotation

    # Further possible discretization (e.g., rounding to integers, mapping to finite field - omitted for simplicity in this basic version)
    # ... For true rigor, need representation theory on discrete groups/finite fields.

    return discrete_rotation_matrix


def apply_discrete_rotation(matrix, discrete_rotation_mat):
    """
    Applies the discrete rotation operator.

    Args:
        matrix (np.ndarray): Matrix to rotate.
        discrete_rotation_mat (np.ndarray): Discrete rotation matrix.

    Returns:
        np.ndarray: Rotated matrix.
    """
    if matrix.shape != discrete_rotation_mat.shape:
        raise ValueError("Matrix and rotation matrix dimensions must match for discrete rotation.")
    rotated_matrix = discrete_rotation_mat @ matrix @ discrete_rotation_mat.T
    return rotated_matrix


def estimate_quantization_effect_modular(matrix, modular_weyl_x, modular_weyl_p):
    """
    Estimates quantization effect using modular Weyl operators (same principle as before).
    """
    matrix_xp = modular_weyl_p(modular_weyl_x(matrix.copy()))
    matrix_px = modular_weyl_x(modular_weyl_p(matrix.copy()))
    difference_matrix = matrix_xp - matrix_px
    quantization_estimate = np.linalg.norm(difference_matrix, 'fro')
    return quantization_estimate


def build_modular_correction_matrix(initial_matrix, rotation_angle_degrees, modular_weyl_x, modular_weyl_p, correction_factor=0.005):
    """
    Builds a modular correction matrix, now incorporating modular Weyl operators
    and quantization estimate derived from the modular setting.
    Lemma system concept is still conceptually applied as in previous version.
    """
    quantization_estimate = estimate_quantization_effect_modular(initial_matrix, modular_weyl_x, modular_weyl_p)
    correction_scale = correction_factor * quantization_estimate
    correction_matrix = np.eye(initial_matrix.shape[0]) * (1.0 - correction_scale) # Size now dynamic

    return correction_matrix


def apply_modular_corrected_rotation(matrix, discrete_rotation_mat, correction_matrix):
    """
    Applies modular corrected rotation using discrete rotation matrix and modular correction.
    """
    if matrix.shape != discrete_rotation_mat.shape or matrix.shape != correction_matrix.shape:
        raise ValueError("Matrix and rotation/correction matrices must have compatible dimensions.")

    corrected_matrix_pre_rotation = correction_matrix @ matrix
    rotated_matrix_corrected = discrete_rotation_mat @ corrected_matrix_pre_rotation @ discrete_rotation_mat.T
    return rotated_matrix_corrected


def demonstrate_modular_weyl_algebra_and_rotation(matrix_dim, modulus, quantization_level, rotation_angle_deg):
    """Demonstrates Weyl algebra with modular space and discrete rotation approximation."""
    print(f"\n--- Demonstrating Modular Weyl Algebra, Discrete Rotation, and Lemma-Based Correction ---")
    print(f"--- (Matrix Dim: {matrix_dim}, Modulus: {modulus}, Quantization Level: {quantization_level}) ---\n")

    initial_matrix = np.eye(matrix_dim)

    print("Initial Matrix:")
    print(initial_matrix)

    modular_space = create_modular_space(matrix_dim, modulus)
    print("\nModular Space (Indices mod {}):".format(modulus))
    print(modular_space)

    modular_weyl_x = modular_weyl_x_operator(matrix_dim, modulus, quantization_level)
    modular_weyl_p = modular_weyl_p_operator(matrix_dim, modulus, quantization_level)

    discrete_rot_mat = discrete_rotation_operator(rotation_angle_deg, matrix_dim) # Discrete rotation approximation
    print(f"\nDiscrete Rotation Matrix ({rotation_angle_deg} degrees - Approximation, Dim: {matrix_dim}x{matrix_dim}):")
    print(discrete_rot_mat)

    # Estimate Quantization Effect (Modular)
    quantization_effect = estimate_quantization_effect_modular(initial_matrix.copy(), modular_weyl_x, modular_weyl_p)
    print(f"\nEstimated Quantization Effect (Modular Weyl Non-commutativity Norm): {quantization_effect:.6f}")

    # Build Modular Correction Matrix (Lemma-Based, Modular Quantization Estimate)
    correction_mat = build_modular_correction_matrix(initial_matrix.copy(), rotation_angle_deg, modular_weyl_x, modular_weyl_p)
    print("\nModular Irrational Factor Correction Matrix (Lemma-Based):")
    print(correction_mat)

    # Apply Rotations
    rotated_matrix_standard_discrete = apply_discrete_rotation(initial_matrix.copy(), discrete_rot_mat)
    rotated_matrix_corrected_modular = apply_modular_corrected_rotation(initial_matrix.copy(), discrete_rot_mat, correction_mat)

    print("\nStandard Discrete Rotated Matrix:")
    print(rotated_matrix_standard_discrete)
    print("\nModular Corrected Discrete Rotated Matrix (Lemma-Based Correction):")
    print(rotated_matrix_corrected_modular)

    difference_corrected_standard = rotated_matrix_corrected_modular - rotated_matrix_standard_discrete
    print("\nDifference between Modular Corrected and Standard Discrete Rotation:")
    print(difference_corrected_standard)
    diff_norm = np.linalg.norm(difference_corrected_standard, 'fro')
    print(f"\nFrobenius Norm of Difference: {diff_norm:.6f}")
    if diff_norm > 1e-9:
        print("\nModular lemma-based correction does alter the discrete rotation result.")
    else:
        print("\nModular lemma-based correction has negligible effect (or correction factor too small).")

    print("\n--- End of Modular Demonstration ---")


if __name__ == "__main__":
    matrix_dimension_modular = 4  # Experiment with dimension
    modulus_value = 4            # Experiment with modulus (try prime numbers for closer to field)
    quantization = 1
    rotation_angle = 45

    demonstrate_modular_weyl_algebra_and_rotation(matrix_dimension_modular, modulus_value, quantization, rotation_angle)
    demonstrate_modular_weyl_algebra_and_rotation(matrix_dimension_modular, modulus_value=5, quantization=1, rotation_angle=45) # Change modulus (prime)
    demonstrate_modular_weyl_algebra_and_rotation(matrix_dimension_modular, modulus_value, quantization=2, rotation_angle=45) # Higher quantization
    demonstrate_modular_weyl_algebra_and_rotation(matrix_dimension_modular, modulus_value, quantization=1, rotation_angle=30) # Different angle
