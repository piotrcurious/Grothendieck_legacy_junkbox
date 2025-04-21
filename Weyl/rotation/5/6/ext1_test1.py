def apply_symbolic_modular_corrected_rotation(matrix: np.ndarray,
                                              rotation_matrix: np.ndarray,
                                              correction_poly_matrix: Matrix) -> np.ndarray:
    """
    Apply symbolic matrix-valued polynomial correction followed by rotation.
    Args:
        matrix (np.ndarray): Original matrix (converted to symbolic).
        rotation_matrix (np.ndarray): Standard numeric rotation.
        correction_poly_matrix (sympy.Matrix): Symbolic polynomial correction matrix.
    Returns:
        np.ndarray: Evaluated corrected rotated matrix at x = 1 (for demonstration).
    """
    from sympy import lambdify

    dim = matrix.shape[0]
    x = symbols('x')

    # Evaluate polynomial matrix at x = 1 for application (demo)
    correction_numeric = correction_poly_matrix.subs(x, 1)
    correction_numeric = np.array(correction_numeric.tolist()).astype(np.float64)

    # Apply correction and rotation numerically
    return rotation_matrix @ (correction_numeric @ matrix) @ rotation_matrix.T

def run_symbolic_rotation_test(shape_type: str,
                               matrix_dim: int,
                               angle_degrees: float,
                               modulus: int,
                               poly_mod_coeffs: list,
                               degree: int = 2):
    """
    Run symbolic rotation with polynomial correction.
    """
    from sympy import Matrix as SymMatrix

    # Create original shape
    shape = create_shape_matrix(shape_type, matrix_dim)
    rotation_matrix = discrete_rotation_operator(angle_degrees, matrix_dim)

    # Modular phase space
    phase_space = ModularPhaseSpace(matrix_dim, modulus)
    weyl_x = ModularWeylXOperator(phase_space)
    weyl_p = ModularWeylPOperator(phase_space)

    # Symbolic polynomial correction strategy
    strategy = PolynomialMatrixCorrectionStrategy(modulus, poly_mod_coeffs, degree=degree)
    correction_poly = strategy.get_correction_matrix(shape, angle_degrees, weyl_x, weyl_p)

    # Apply standard and corrected rotations
    rotated_standard = apply_discrete_rotation(shape, rotation_matrix)
    rotated_corrected = apply_symbolic_modular_corrected_rotation(shape, rotation_matrix, correction_poly)

    # Calculate metrics
    metrics = calculate_rotation_quality(shape, rotated_standard, rotated_corrected)
    print(f"Test: Shape={shape_type}, Angle={angle_degrees}°")
    print("Overlap (Standard):", metrics['overlap_standard'])
    print("Overlap (Corrected):", metrics['overlap_corrected'])
    print("Frobenius norm diff:", metrics['difference_norm'])

    # Optional plot
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(shape, cmap='gray')
    axs[0].set_title('Original')
    axs[1].imshow(rotated_standard, cmap='inferno')
    axs[1].set_title('Standard Rotated')
    axs[2].imshow(rotated_corrected, cmap='viridis')
    axs[2].set_title('Corrected Rotated')
    plt.tight_layout()
    plt.show()

    run_symbolic_rotation_test(
    shape_type='circle',
    matrix_dim=16,
    angle_degrees=35,
    modulus=7,
    poly_mod_coeffs=[1, 0, 1, 1],  # x^3 + x + 1
    degree=2
    )
