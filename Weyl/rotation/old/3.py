import numpy as np

class ModularPhaseSpace:
    """
    Represents a discrete phase space based on integers modulo 'modulus'.

    This class encapsulates the modular arithmetic and index handling for operators
    acting in this discretized space.  Using a class improves abstraction and organization.

    Args:
        dimension (int): Dimension of the space (and matrices operating on it).
        modulus (int): Modulus for integer arithmetic (defining Z/modulus Z).
    """
    def __init__(self, dimension, modulus):
        self.dimension = dimension
        self.modulus = modulus
        if modulus <= 0:
            raise ValueError("Modulus must be a positive integer.")
        self.indices = np.arange(dimension) % modulus # Pre-calculate indices in modular space

    def get_index(self, index):
        """Returns index in modular space (index mod modulus)."""
        return index % self.modulus

    def shift_index(self, index, shift):
        """Shifts an index in the modular space."""
        return (index + shift) % self.modulus

    def get_modular_indices(self):
        """Returns the array of modular indices."""
        return self.indices

    def __repr__(self):
        return f"ModularPhaseSpace(dimension={self.dimension}, modulus={self.modulus})"


class Operator:
    """
    Abstract base class for operators in the discrete phase space.
    Provides a template for operator classes and enforces consistent interface.
    """
    def __init__(self, phase_space):
        if not isinstance(phase_space, ModularPhaseSpace):
            raise TypeError("Operator must be initialized with a ModularPhaseSpace object.")
        self.phase_space = phase_space

    def operate(self, matrix):
        """
        Abstract method to apply the operator to a matrix.
        Subclasses must implement this method.
        """
        raise NotImplementedError("Subclasses must implement the operate method.")

    def __call__(self, matrix): # Allows operator instances to be called as functions: op(matrix)
        return self.operate(matrix)


class ModularWeylXOperator(Operator):
    """
    Modular Weyl X operator, now inheriting from Operator base class and using ModularPhaseSpace.
    """
    def __init__(self, phase_space, quantization_level=1):
        super().__init__(phase_space) # Initialize base class
        self.quantization_level = quantization_level

    def operate(self, matrix):
        if not isinstance(matrix, np.ndarray) or matrix.shape != (self.phase_space.dimension, self.phase_space.dimension):
            raise ValueError("Input must be a square NumPy matrix of specified dimension.")

        shifted_matrix = np.zeros_like(matrix)
        for i in range(self.phase_space.dimension):
            shifted_row_index = self.phase_space.shift_index(i, self.quantization_level) # Use ModularPhaseSpace for shift
            shifted_matrix[shifted_row_index, :] = matrix[i, :]
        return shifted_matrix

class ModularWeylPOperator(Operator):
    """
    Modular Weyl P operator, inheriting from Operator and using ModularPhaseSpace.
    """
    def __init__(self, phase_space, quantization_level=1):
        super().__init__(phase_space)
        self.quantization_level = quantization_level

    def operate(self, matrix):
        if not isinstance(matrix, np.ndarray) or matrix.shape != (self.phase_space.dimension, self.phase_space.dimension):
            raise ValueError("Input must be a square NumPy matrix of specified dimension.")

        shifted_matrix = np.zeros_like(matrix)
        for j in range(self.phase_space.dimension):
            shifted_col_index = self.phase_space.shift_index(j, self.quantization_level) # Use ModularPhaseSpace for shift
            shifted_matrix[:, shifted_col_index] = matrix[:, j]
        return shifted_matrix


def discrete_rotation_operator(angle_degrees, discrete_space_dimension):
    """
    Discrete Rotation Operator (Approximation) -  Improved Documentation for Rigor.

    As noted in previous versions, truly rigorous rotation in a discrete phase space
    is mathematically complex and requires advanced techniques from representation theory
    on discrete groups or finite fields.

    This function provides a *simplified approximation* by:
    1. Constructing a standard 2D rotation matrix in continuous space (using floats).
    2. Embedding this 2D rotation into a larger identity matrix.
    3. (Optional, for future rigor) One could consider discretizing the entries further
       or using more sophisticated methods from discrete mathematics to represent rotations
       more accurately in Z_modulus Z.

    Limitations of this approximation should be kept in mind when interpreting results.
    For demonstrations focusing on Weyl algebra and quantization effects, this approximation
    can still be illustrative, but for applications requiring precise discrete rotations,
    more advanced methods would be necessary.

    Args:
        angle_degrees (float): Rotation angle.
        discrete_space_dimension (int): Dimension of the discrete space.

    Returns:
        np.ndarray: Discrete rotation matrix (approximation).
    """
    angle_radians = np.radians(angle_degrees)
    continuous_rotation_matrix_2d = np.array([[np.cos(angle_radians), -np.sin(angle_radians)],
                                              [np.sin(angle_radians),  np.cos(angle_radians)]])

    discrete_rotation_matrix = np.eye(discrete_space_dimension, dtype=float)
    if discrete_space_dimension >= 2:
        discrete_rotation_matrix[0:2, 0:2] = continuous_rotation_matrix_2d

    return discrete_rotation_matrix


def apply_discrete_rotation(matrix, discrete_rotation_mat):
    """Applies the discrete rotation operator (same as before)."""
    if matrix.shape != discrete_rotation_mat.shape:
        raise ValueError("Matrix and rotation matrix dimensions must match for discrete rotation.")
    rotated_matrix = discrete_rotation_mat @ matrix @ discrete_rotation_mat.T
    return rotated_matrix


def estimate_quantization_effect_modular(matrix, modular_weyl_x, modular_weyl_p):
    """Estimates quantization effect using modular Weyl operators (same as before)."""
    matrix_xp = modular_weyl_p(modular_weyl_x(matrix.copy()))
    matrix_px = modular_weyl_x(modular_weyl_p(matrix.copy()))
    difference_matrix = matrix_xp - matrix_px
    quantization_estimate = np.linalg.norm(difference_matrix, 'fro')
    return quantization_estimate


class CorrectionStrategy: # Abstract base for correction strategies (for future extension)
    """Abstract base class for correction strategies."""
    def __init__(self):
        pass
    def get_correction_matrix(self, initial_matrix, rotation_angle_degrees, weyl_x, weyl_p):
        """Abstract method to get the correction matrix."""
        raise NotImplementedError("Subclasses must implement get_correction_matrix.")


class LemmaBasedCorrectionStrategy(CorrectionStrategy):
    """
    Lemma-Based Correction Strategy, now as a class inheriting from CorrectionStrategy.
    Encapsulates the lemma-based correction logic, improving abstraction.
    """
    def __init__(self, correction_factor=0.005):
        super().__init__() # Initialize base class
        self.correction_factor = correction_factor

    def get_correction_matrix(self, initial_matrix, rotation_angle_degrees, modular_weyl_x, modular_weyl_p):
        """
        Builds a modular correction matrix using the lemma system (as before).
        Now part of a CorrectionStrategy class for better organization.
        """
        quantization_estimate = estimate_quantization_effect_modular(initial_matrix, modular_weyl_x, modular_weyl_p)
        correction_scale = self.correction_factor * quantization_estimate
        correction_matrix = np.eye(initial_matrix.shape[0]) * (1.0 - correction_scale)
        return correction_matrix


def apply_modular_corrected_rotation(matrix, discrete_rotation_mat, correction_matrix):
    """Applies modular corrected rotation (same as before)."""
    if matrix.shape != discrete_rotation_mat.shape or matrix.shape != correction_matrix.shape:
        raise ValueError("Matrix and rotation/correction matrices must have compatible dimensions.")

    corrected_matrix_pre_rotation = correction_matrix @ matrix
    rotated_matrix_corrected = discrete_rotation_mat @ corrected_matrix_pre_rotation @ discrete_rotation_mat.T
    return rotated_matrix_corrected


def demonstrate_modular_weyl_algebra_and_rotation(matrix_dim, modulus, quantization_level, rotation_angle_deg, correction_strategy=None):
    """Demonstrates modular Weyl algebra, discrete rotation, and correction using Strategy Pattern."""
    print(f"\n--- Demonstrating Modular Weyl Algebra, Discrete Rotation, and Correction Strategy ---")
    print(f"--- (Matrix Dim: {matrix_dim}, Modulus: {modulus}, Quantization Level: {quantization_level}) ---\n")

    initial_matrix = np.eye(matrix_dim)

    print("Initial Matrix:")
    print(initial_matrix)

    phase_space = ModularPhaseSpace(matrix_dim, modulus) # Create ModularPhaseSpace instance
    print("\nModular Phase Space (Indices mod {}):".format(modulus))
    print(phase_space.get_modular_indices())

    modular_weyl_x = ModularWeylXOperator(phase_space, quantization_level) # Operators now take PhaseSpace
    modular_weyl_p = ModularWeylPOperator(phase_space, quantization_level)

    discrete_rot_mat = discrete_rotation_operator(rotation_angle_deg, matrix_dim)
    print(f"\nDiscrete Rotation Matrix ({rotation_angle_deg} degrees - Approximation, Dim: {matrix_dim}x{matrix_dim}):")
    print(discrete_rot_mat)

    # Estimate Quantization Effect (Modular, using PhaseSpace in operators implicitly)
    quantization_effect = estimate_quantization_effect_modular(initial_matrix.copy(), modular_weyl_x, modular_weyl_p)
    print(f"\nEstimated Quantization Effect (Modular Weyl Non-commutativity Norm): {quantization_effect:.6f}")

    correction_mat = np.eye(matrix_dim) # Default: No correction (Identity matrix)
    if correction_strategy: # Use correction strategy if provided
        if not isinstance(correction_strategy, CorrectionStrategy):
            raise TypeError("correction_strategy must be an instance of CorrectionStrategy.")
        correction_mat = correction_strategy.get_correction_matrix(initial_matrix.copy(), rotation_angle_deg, modular_weyl_x, modular_weyl_p)
        print("\nModular Irrational Factor Correction Matrix (Strategy-Based):")
        print(correction_mat)
    else:
        print("\nNo Correction Strategy Applied (Using Identity Correction Matrix).")


    rotated_matrix_standard_discrete = apply_discrete_rotation(initial_matrix.copy(), discrete_rot_mat)
    rotated_matrix_corrected_modular = apply_modular_corrected_rotation(initial_matrix.copy(), discrete_rot_mat, correction_mat)

    print("\nStandard Discrete Rotated Matrix:")
    print(rotated_matrix_standard_discrete)
    print("\nCorrected Discrete Rotated Matrix (Strategy-Based Correction):")
    print(rotated_matrix_corrected_modular)

    difference_corrected_standard = rotated_matrix_corrected_modular - rotated_matrix_standard_discrete
    print("\nDifference between Corrected and Standard Discrete Rotation:")
    print(difference_corrected_standard)
    diff_norm = np.linalg.norm(difference_corrected_standard, 'fro')
    print(f"\nFrobenius Norm of Difference: {diff_norm:.6f}")
    if diff_norm > 1e-9:
        print("\nStrategy-based correction alters the discrete rotation result.")
    else:
        print("\nStrategy-based correction has negligible effect (or factor too small/strategy ineffective).")

    print("\n--- End of Modular Demonstration ---")


if __name__ == "__main__":
    matrix_dimension_modular = 4
    modulus_value = 5
    quantization = 1
    rotation_angle = 45

    lemma_correction = LemmaBasedCorrectionStrategy(correction_factor=0.005) # Create a correction strategy instance

    demonstrate_modular_weyl_algebra_and_rotation(matrix_dimension_modular, modulus_value, quantization, rotation_angle, correction_strategy=lemma_correction) # Pass strategy
    demonstrate_modular_weyl_algebra_and_rotation(matrix_dimension_modular, modulus_value, quantization=2, rotation_angle=45, correction_strategy=lemma_correction) # Higher quantization
    demonstrate_modular_weyl_algebra_and_rotation(matrix_dimension_modular, modulus_value, quantization=1, rotation_angle=30, correction_strategy=lemma_correction) # Different angle
    demonstrate_modular_weyl_algebra_and_rotation(matrix_dimension_modular, modulus_value, quantization=1, rotation_angle=75, correction_strategy=lemma_correction) # Another angle
    demonstrate_modular_weyl_algebra_and_rotation(matrix_dimension_modular, modulus_value, quantization, rotation_angle, correction_strategy=None) # No correction strategy
