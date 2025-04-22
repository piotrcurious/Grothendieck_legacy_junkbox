import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Type
# Required for correct image-like rotation:
try:
    import scipy.ndimage
except ImportError:
    print("Error: SciPy is required for correct rotation. Please install it: pip install scipy")
    exit()

# Type alias for NumPy arrays
NDArray = np.ndarray

##############################################
# Core Mathematical Structures and Operators
# (P Operator Updated)
##############################################

class ModularPhaseSpace:
    """Represents a discrete phase space isomorphic to Z_modulus."""
    def __init__(self, dimension: int, modulus: int) -> None:
        if not isinstance(dimension, int) or dimension <= 0: raise ValueError("Dimension must be positive.")
        if not isinstance(modulus, int) or modulus <= 0: raise ValueError("Modulus must be positive.")
        self.dimension = dimension; self.modulus = modulus
    def get_index(self, index: int) -> int: return index % self.modulus
    def shift_index(self, index: int, shift: int) -> int: return (index + shift) % self.modulus
    def __repr__(self) -> str: return f"ModularPhaseSpace(dimension={self.dimension}, modulus={self.modulus})"

class Operator(ABC):
    """Abstract base class for operators acting on matrices."""
    def __init__(self, phase_space: ModularPhaseSpace) -> None:
        if not isinstance(phase_space, ModularPhaseSpace): raise TypeError("Operator must use ModularPhaseSpace.")
        self.phase_space = phase_space
    @abstractmethod
    def operate(self, matrix: NDArray) -> NDArray: pass
    def __call__(self, matrix: NDArray) -> NDArray: return self.operate(matrix)
    def __repr__(self) -> str:
        params = [f"{k}={v}" for k, v in self.__dict__.items() if hasattr(self, k) and k != 'phase_space' and not k.startswith('_')]
        return f"{self.__class__.__name__}({', '.join(params)})"

class ModularWeylXOperator(Operator):
    """Modular Weyl X (Position Shift) operator using np.roll for cyclic row shifts."""
    def __init__(self, phase_space: ModularPhaseSpace, quantization_level: int = 1) -> None:
        super().__init__(phase_space)
        self.quantization_level = quantization_level % self.phase_space.modulus
    def operate(self, matrix: NDArray) -> NDArray:
        """Applies cyclic shift to rows."""
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]: raise ValueError("Matrix must be square.")
        if matrix.shape[0] != self.phase_space.dimension: raise ValueError(f"Matrix dim {matrix.shape[0]} != phase space dim {self.phase_space.dimension}.")
        # Positive shift moves elements "down" (higher row indices)
        return np.roll(matrix, shift=self.quantization_level, axis=0)

class ModularWeylPOperator(Operator):
    """
    Modular Weyl P (Momentum Shift / Phase) operator.
    Multiplies rows by phase factors based on row index.
    (P * M)[i, j] = omega^(i * level) * M[i, j]
    where omega = exp(2*pi*i / modulus).
    """
    def __init__(self, phase_space: ModularPhaseSpace, quantization_level: int = 1) -> None:
        super().__init__(phase_space)
        self.quantization_level = quantization_level # Keep original level for phase calculation
        # Precompute phase factors for efficiency
        omega = np.exp(2j * np.pi / self.phase_space.modulus)
        row_indices = np.arange(self.phase_space.dimension)
        # Phase factor for each row: omega^(i * level)
        self.phase_factors = omega**(row_indices * self.quantization_level)
        # Reshape for broadcasting: (dim, 1) to multiply rows element-wise
        self.phase_factors = self.phase_factors[:, np.newaxis]

    def operate(self, matrix: NDArray) -> NDArray:
        """Applies phase multiplication based on row index."""
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]: raise ValueError("Matrix must be square.")
        if matrix.shape[0] != self.phase_space.dimension: raise ValueError(f"Matrix dim {matrix.shape[0]} != phase space dim {self.phase_space.dimension}.")

        # Ensure matrix is complex type for multiplication
        matrix_complex = matrix.astype(np.complex128, copy=False)

        # Multiply rows by precomputed phase factors using broadcasting
        return self.phase_factors * matrix_complex

def estimate_quantization_effect_modular(matrix: NDArray, weyl_x: Operator, weyl_p: Operator) -> float:
    """
    Estimates quantization effect norm: Q = || P(X(matrix)) - X(P(matrix)) ||_F.
    Handles complex matrices resulting from the phase operator P.
    """
    # Apply X then P
    x_matrix = weyl_x(matrix)
    p_of_x = weyl_p(x_matrix) # Result will be complex

    # Apply P then X
    p_matrix = weyl_p(matrix) # Result will be complex
    x_of_p = weyl_x(p_matrix) # Shifting complex matrix

    # Difference will be complex
    difference = p_of_x - x_of_p
    # Frobenius norm handles complex numbers correctly
    norm_q = float(np.linalg.norm(difference, 'fro'))
    return norm_q


##############################################
# Correction Strategies
# (Using revised Lemma scaling from V7)
##############################################

class CorrectionStrategy(ABC):
    """Abstract base class for correction strategies."""
    @abstractmethod
    def get_correction_matrix(self, matrix: NDArray, angle_deg: float,
                              weyl_x: Operator, weyl_p: Operator) -> NDArray: pass
    def __repr__(self) -> str:
        params = [f"{k}={v}" for k, v in self.__dict__.items() if hasattr(self, k) and k != 'phase_space' and not k.startswith('_')]
        return f"{self.__class__.__name__}({', '.join(params)})"

class LemmaBasedCorrectionStrategy(CorrectionStrategy):
    """
    Correction: Scales identity matrix based on Q = ||PX - XP||.
    Uses scaling: scale = 1.0 / (1.0 + factor * Q).
    """
    def __init__(self, correction_factor: float = 0.01) -> None:
        if correction_factor < 0: print(f"Warning: Negative correction_factor ({correction_factor}).")
        self.correction_factor = correction_factor
        self._last_q = None; self._last_scale = None

    def get_correction_matrix(self, matrix: NDArray, angle_deg: float, weyl_x: Operator, weyl_p: Operator) -> NDArray:
        """Calculates Correction = Identity * (1 / (1 + factor * Q))."""
        # Q should now be non-zero with the new P operator
        Q = estimate_quantization_effect_modular(matrix, weyl_x, weyl_p)
        scale = 1.0 / (1.0 + self.correction_factor * Q)
        self._last_q = Q; self._last_scale = scale
        # Correction matrix C is real and diagonal
        return np.eye(matrix.shape[0], dtype=float) * scale

    def get_last_q_value(self) -> Optional[float]: return self._last_q
    def get_last_scale_factor(self) -> Optional[float]: return self._last_scale

class RandomNoiseCorrectionStrategy(CorrectionStrategy):
    """Correction: Adds random noise to the identity matrix."""
    def __init__(self, noise_factor: float = 0.01) -> None:
        self.noise_factor = noise_factor
    def get_correction_matrix(self, matrix: NDArray, angle_deg: float, weyl_x: Operator, weyl_p: Operator) -> NDArray:
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]: raise ValueError("Matrix must be square 2D.")
        dim = matrix.shape[0]; noise = (np.random.rand(dim, dim) - 0.5) * self.noise_factor
        # Correction matrix C is real
        return np.eye(dim, dtype=float) + noise

##############################################
# Rotation and Application
# (Unchanged - handles real C and matrix)
##############################################

def rotate_matrix_around_center(matrix: NDArray, angle_degrees: float) -> NDArray:
    """Rotates a 2D matrix around its center using scipy.ndimage.rotate."""
    if matrix.ndim != 2: raise ValueError("Input matrix must be 2D.")
    # Input matrix might be complex if P was applied, but rotation handles it.
    # However, in the current flow, rotate is applied *after* C, and C@matrix is real.
    return scipy.ndimage.rotate(matrix, angle=angle_degrees, reshape=False, mode='constant', cval=0.0, order=1)

def apply_corrected_rotation(matrix: NDArray, C: NDArray, angle_degrees: float) -> NDArray:
    """Applies correction C then rotates: Result = Rotate(C @ matrix)."""
    if not (matrix.shape == C.shape): raise ValueError(f"Shapes must match. Got matrix: {matrix.shape}, C: {C.shape}")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]: raise ValueError("Inputs must be square 2D arrays.")
    # C is real (from both strategies), matrix is real. C @ matrix is real.
    corrected_matrix = C @ matrix
    rotated_corrected_matrix = rotate_matrix_around_center(corrected_matrix, angle_degrees)
    return rotated_corrected_matrix

##############################################
# Helper Functions
# (Unchanged)
##############################################

def create_shape_matrix(shape_type: str, dim: int) -> NDArray:
    """Creates a square matrix representing a simple geometric shape centered."""
    # (Implementation unchanged)
    if dim <= 0: raise ValueError("Dimension must be positive.")
    mat = np.zeros((dim, dim), dtype=float); cx, cy = dim // 2, dim // 2
    if shape_type == 'square':
        size = max(1, dim // 3); start = cx - size // 2; end = start + size
        row_slice = slice(max(0, start), min(dim, end)); col_slice = slice(max(0, start), min(dim, end))
        mat[row_slice, col_slice] = 1.0
    elif shape_type == 'circle':
        radius_sq = (dim / 4.0)**2; Y, X = np.ogrid[:dim, :dim]
        dist_sq = (X - cx)**2 + (Y - cy)**2; mask = dist_sq <= radius_sq; mat[mask] = 1.0
    elif shape_type == 'triangle':
        base_half_width = dim // 4; apex_y = cy // 2
        if dim > 1 and (dim - 1 - apex_y) > 0:
            for i in range(apex_y, dim):
                progress = (i - apex_y) / (dim - 1 - apex_y); half_width_at_row = int(base_half_width * progress)
                j_start = cx - half_width_at_row; j_end = cx + half_width_at_row + 1
                mat[i, max(0, j_start):min(dim, j_end)] = 1.0
        elif dim == 1: mat[0,0] = 1.0
    elif shape_type == 'line':
        line_thickness = max(1, dim // 16); row_start = cy - line_thickness // 2; row_end = row_start + line_thickness
        col_start = dim // 4; col_end = 3 * dim // 4
        mat[max(0, row_start):min(dim, row_end), max(0, col_start):min(dim, col_end)] = 1.0
    elif shape_type == 'cross':
        arm_thickness = max(1, dim // 10); center_offset = arm_thickness // 2
        row_start_h = cy - center_offset; row_end_h = row_start_h + arm_thickness
        mat[max(0, row_start_h):min(dim, row_end_h), :] = 1.0
        col_start_v = cx - center_offset; col_end_v = col_start_v + arm_thickness
        mat[:, max(0, col_start_v):min(dim, col_end_v)] = 1.0
    else: raise ValueError(f"Unsupported shape_type: '{shape_type}'.")
    return mat

def calculate_rotation_quality(original: NDArray, standard_rotated: NDArray, corrected_rotated: NDArray) -> Dict[str, float]:
    """Calculates metrics: overlap and difference norm."""
    # (Implementation unchanged)
    if not (original.shape == standard_rotated.shape == corrected_rotated.shape): raise ValueError("Matrices must have same shape.")
    if original.ndim != 2: raise ValueError("Matrices must be 2D.")
    overlap_std = float(np.sum(original * standard_rotated)); overlap_corr = float(np.sum(original * corrected_rotated))
    diff_norm = float(np.linalg.norm(corrected_rotated - standard_rotated, 'fro'))
    return {'overlap_standard': overlap_std, 'overlap_corrected': overlap_corr, 'difference_norm': diff_norm}

##############################################
# Visualization Functions
# (Updated display_weyl_commutation for complex)
##############################################

def display_weyl_commutation(matrix: NDArray, weyl_x: Operator, weyl_p: Operator,
                             shape_name: str) -> plt.Figure:
    """
    Visualizes the non-commutativity of Weyl operators P and X.
    Displays P(X(matrix)), X(P(matrix)), and their difference.
    Handles complex matrices from the phase operator P.
    """
    print(f"--- Visualizing Weyl Commutation for Shape: {shape_name} ---")
    # Calculate the terms
    x_matrix = weyl_x(matrix)
    p_of_x = weyl_p(x_matrix) # Complex

    p_matrix = weyl_p(matrix) # Complex
    x_of_p = weyl_x(p_matrix) # Complex

    difference = p_of_x - x_of_p # Complex
    norm_q = float(np.linalg.norm(difference, 'fro'))
    print(f"  ||P(X(matrix)) - X(P(matrix))||_F = {norm_q:.4f}") # Should be > 0 now

    # Visualization setup
    fig, axes = plt.subplots(1, 3, figsize=(18, 7)) # Slightly taller for titles
    # Use magnitude for display, as phase is hard to visualize directly with imshow
    mag_px = np.abs(p_of_x)
    mag_xp = np.abs(x_of_p)
    mag_diff = np.abs(difference)
    vmax_mag = max(mag_px.max(), mag_xp.max(), 1e-6)
    vmax_diff = max(mag_diff.max(), 1e-6)

    # Plot Magnitude of P(X(matrix))
    im0 = axes[0].imshow(mag_px, cmap='viridis', interpolation='nearest', vmin=0, vmax=vmax_mag)
    axes[0].set_title("|P(X(matrix))|")
    axes[0].axis('off'); axes[0].set_aspect('equal')
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Plot Magnitude of X(P(matrix))
    im1 = axes[1].imshow(mag_xp, cmap='viridis', interpolation='nearest', vmin=0, vmax=vmax_mag)
    axes[1].set_title("|X(P(matrix))|")
    axes[1].axis('off'); axes[1].set_aspect('equal')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Plot Magnitude of Difference
    im2 = axes[2].imshow(mag_diff, cmap='magma', interpolation='nearest', vmin=0, vmax=vmax_diff)
    axes[2].set_title(f"|PX - XP|\nNorm Q = {norm_q:.3f}")
    axes[2].axis('off'); axes[2].set_aspect('equal')
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(f"Weyl Operator Non-Commutativity (Magnitude Display)\nShape: {shape_name.capitalize()}, Quant Level: {weyl_x.quantization_level}", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.90])
    return fig


def display_matrices_graphical(original: NDArray, standard_rotated: NDArray, corrected_rotated: NDArray,
                               title_prefix: str, angle: float, quant_level: int,
                               strategy_info: str, modulus: int,
                               metrics: Dict[str, float],
                               quantization_effect_norm: Optional[float] = None) -> plt.Figure:
    """Displays original, standard rotated, and corrected rotated matrices."""
    # (Implementation unchanged from V5)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    vmax = max(original.max(), standard_rotated.max(), corrected_rotated.max(), 1e-6); vmin = 0
    im0 = axes[0].imshow(original, cmap='gray', interpolation='nearest', vmin=vmin, vmax=vmax)
    axes[0].set_title("Original Shape"); axes[0].axis('off'); axes[0].set_aspect('equal')
    im1 = axes[1].imshow(standard_rotated, cmap='gray', interpolation='nearest', vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Standard Rotation\nOverlap w/ Original: {metrics['overlap_standard']:.3f}"); axes[1].axis('off'); axes[1].set_aspect('equal')
    im2 = axes[2].imshow(corrected_rotated, cmap='gray', interpolation='nearest', vmin=vmin, vmax=vmax)
    axes[2].set_title(f"Corrected Rotation ({strategy_info})\nOverlap w/ Original: {metrics['overlap_corrected']:.3f}\nDiff Norm from Std: {metrics['difference_norm']:.3f}"); axes[2].axis('off'); axes[2].set_aspect('equal')
    main_title = f"{title_prefix}\nAngle: {angle:.1f}°, Quant Level: {quant_level}, Modulus: {modulus}"
    if quantization_effect_norm is not None: main_title += f"\nQuantization Effect Norm (Q) = {quantization_effect_norm:.3f}"
    fig.suptitle(main_title, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.90]); return fig

def run_experiment_parameter_variation(*args, **kwargs):
    """Runs rotation experiments varying one parameter, collecting metrics."""
    # (No changes needed in the function logic itself)
    dim = kwargs['dim']; modulus = kwargs['modulus']; correction_strategy = kwargs['correction_strategy']
    param_name = kwargs['param_name']; param_values = kwargs['param_values']; shape_type = kwargs['shape_type']
    fixed_quantization_level = kwargs['fixed_quantization_level']; fixed_rotation_angle_deg = kwargs['fixed_rotation_angle_deg']
    if param_name not in ['quantization_level', 'rotation_angle_deg']: raise ValueError("param_name invalid")
    results = []; phase_space = ModularPhaseSpace(dim, modulus); initial_matrix = create_shape_matrix(shape_type, dim); strategy_name = repr(correction_strategy)
    print(f"Running experiment varying '{param_name}' for shape '{shape_type}' over {len(param_values)} values...")
    for i, val in enumerate(param_values):
        if param_name == 'quantization_level': current_quant_level = int(val); current_angle_deg = fixed_rotation_angle_deg
        else: current_quant_level = fixed_quantization_level; current_angle_deg = float(val)
        # Use updated P operator definition
        weyl_x = ModularWeylXOperator(phase_space, current_quant_level); weyl_p = ModularWeylPOperator(phase_space, current_quant_level)
        standard_rotated = rotate_matrix_around_center(initial_matrix, current_angle_deg)
        correction_matrix = correction_strategy.get_correction_matrix(initial_matrix, current_angle_deg, weyl_x, weyl_p)
        corrected_rotated = apply_corrected_rotation(initial_matrix, correction_matrix, current_angle_deg)
        metrics = calculate_rotation_quality(initial_matrix, standard_rotated, corrected_rotated)
        metrics[param_name] = val; metrics['quantization_level'] = current_quant_level; metrics['rotation_angle_deg'] = current_angle_deg
        metrics['strategy'] = strategy_name; metrics['modulus'] = modulus; metrics['dimension'] = dim; metrics['shape_type'] = shape_type
        results.append(metrics)
    print("Experiment finished.")
    return results

def plot_experiment_results(*args, **kwargs):
    """Plots results from parameter variation experiment."""
    # (No changes needed in the function logic itself)
    results = args[0]; param_name = args[1]
    if not results: raise ValueError("Results list is empty.")
    first_result = results[0]; modulus = first_result['modulus']; strategy_name = first_result['strategy']; shape_type = first_result['shape_type']
    fixed_param_info = ""
    if param_name == 'rotation_angle_deg': fixed_param_info = f"Quant Level = {first_result['quantization_level']}"
    elif param_name == 'quantization_level': fixed_param_info = f"Angle = {first_result['rotation_angle_deg']:.1f}°"
    param_vals = [r[param_name] for r in results]; overlap_std = [r['overlap_standard'] for r in results]; overlap_corr = [r['overlap_corrected'] for r in results]; diff_norm = [r['difference_norm'] for r in results]
    fig, ax1 = plt.subplots(figsize=(12, 7)); color1 = 'tab:blue'
    ax1.set_xlabel(f'{param_name.replace("_", " ").title()}'); ax1.set_ylabel('Overlap with Original', color=color1)
    line1, = ax1.plot(param_vals, overlap_std, 'o-', color=color1, label='Standard Rotation Overlap')
    line2, = ax1.plot(param_vals, overlap_corr, 's-', color='tab:cyan', label='Corrected Rotation Overlap')
    ax1.tick_params(axis='y', labelcolor=color1); ax1.grid(True, axis='y', linestyle=':', linewidth=0.5)
    ax2 = ax1.twinx(); color2 = 'tab:red'
    ax2.set_ylabel('Norm(Corrected - Standard)', color=color2)
    line3, = ax2.plot(param_vals, diff_norm, '^--', color=color2, label='Difference Norm')
    ax2.tick_params(axis='y', labelcolor=color2)
    plt.title(f'Rotation Quality vs {param_name.replace("_", " ").title()}\n(Shape: {shape_type}, Mod: {modulus}, {fixed_param_info}, Strategy: {strategy_name})', fontsize=12)
    lines = [line1, line2, line3]; labels = [l.get_label() for l in lines]; ax1.legend(lines, labels, loc='best')
    fig.tight_layout(); plt.show()


# Guard execution
if __name__ == "__main__":

    # --- Experiment Configuration ---
    DIM = 64
    MODULUS = 64 # Should match DIM for phase operator definition
    QUANT_LEVEL = 3 # Try different levels (e.g., 1, 2, 3, 5...)
    ANGLE_DEG = 45.0
    SHAPES_TO_TEST = ['circle', 'square', 'triangle', 'line', 'cross']

    # --- Correction Strategy Setup ---
    # <<< Tune LEMMA_CORR_FACTOR with the new P operator and scaling! >>>
    # Suggestions: 0.001, 0.01, 0.1, 1.0? The sensitivity will be different.
    LEMMA_CORR_FACTOR = 0.01
    NOISE_CORR_FACTOR = 0.05

    # --- Choose Correction Strategy ---
    strategy_instance = LemmaBasedCorrectionStrategy(correction_factor=LEMMA_CORR_FACTOR)
    # strategy_instance = RandomNoiseCorrectionStrategy(noise_factor=NOISE_CORR_FACTOR)

    print(f"Using Strategy: {strategy_instance}")
    print("-" * 30)

    # ==========================================================
    # --- 0. Demonstrate Weyl Commutation Effect (Once) ---
    # ==========================================================
    print(f"--- Demonstrating Weyl Commutation (Quant={QUANT_LEVEL}) ---")
    demo_shape = SHAPES_TO_TEST[0]
    phase_space_demo = ModularPhaseSpace(DIM, MODULUS)
    # Use updated P operator
    weyl_x_demo = ModularWeylXOperator(phase_space_demo, QUANT_LEVEL)
    weyl_p_demo = ModularWeylPOperator(phase_space_demo, QUANT_LEVEL)
    initial_matrix_demo = create_shape_matrix(demo_shape, DIM)
    fig_commutation = display_weyl_commutation(initial_matrix_demo, weyl_x_demo, weyl_p_demo, demo_shape)
    plt.show()
    print("-" * 30)


    # ==========================================================
    # --- 1. Test Different Shapes (Fixed Angle/Quant Level) ---
    # ==========================================================
    print(f"--- Testing Various Shapes (Angle={ANGLE_DEG}°, Quant={QUANT_LEVEL}) ---")
    phase_space_shape_test = phase_space_demo
    weyl_x_shape_test = weyl_x_demo
    weyl_p_shape_test = weyl_p_demo # Use the P operator defined above

    for current_shape in SHAPES_TO_TEST:
        print(f"  Testing Shape: {current_shape}")
        initial_matrix = create_shape_matrix(current_shape, DIM)

        # --- Calculate Q and Scale Factor for Diagnostics (if Lemma strategy) ---
        q_norm_for_display = None
        scale_factor_for_display = None
        if isinstance(strategy_instance, LemmaBasedCorrectionStrategy):
             # Q should now be > 0
             q_norm_for_display = estimate_quantization_effect_modular(initial_matrix, weyl_x_shape_test, weyl_p_shape_test)
             scale_factor_for_display = strategy_instance.get_last_scale_factor() # Get from last calculation
             if scale_factor_for_display is None: # Should have been calculated in get_correction_matrix
                 scale_factor_for_display = 1.0 / (1.0 + strategy_instance.correction_factor * q_norm_for_display)

             print(f"    Lemma Strategy Diagnostics:")
             print(f"      Quantization Effect Norm (Q) = {q_norm_for_display:.4f}") # Expect > 0
             print(f"      Calculated Scale Factor      = {scale_factor_for_display:.4f}") # Expect < 1

        # Standard Rotation
        standard_rotated = rotate_matrix_around_center(initial_matrix, ANGLE_DEG)

        # Corrected Rotation
        correction_matrix = strategy_instance.get_correction_matrix(
            initial_matrix, ANGLE_DEG, weyl_x_shape_test, weyl_p_shape_test
        )
        corrected_rotated = apply_corrected_rotation(
            initial_matrix, correction_matrix, ANGLE_DEG
        )

        # Metrics & Visualization
        quality_metrics = calculate_rotation_quality(initial_matrix, standard_rotated, corrected_rotated)
        fig_shape = display_matrices_graphical(
            original=initial_matrix, standard_rotated=standard_rotated, corrected_rotated=corrected_rotated,
            title_prefix=f"Shape: {current_shape.capitalize()}", angle=ANGLE_DEG, quant_level=QUANT_LEVEL,
            strategy_info=repr(strategy_instance), modulus=MODULUS, metrics=quality_metrics,
            quantization_effect_norm=q_norm_for_display
        )
        plt.show()
        print(f"  Metrics for {current_shape}: {quality_metrics}") # Expect difference_norm > 0 now
        print("-" * 10)
    print("--- Shape Testing Finished ---")
    print("-" * 30)


    # =====================================================
    # --- 2. Parameter Variation Example: Varying Angle ---
    # =====================================================
    # (Code unchanged)
    shape_for_param_sweep = SHAPES_TO_TEST[0]
    print(f"\n--- Running Parameter Variation (Varying Angle, Shape='{shape_for_param_sweep}') ---")
    angle_values_to_test = np.linspace(0, 180, 19)
    results_angle_variation = run_experiment_parameter_variation(
        dim=DIM, modulus=MODULUS, fixed_quantization_level=QUANT_LEVEL, fixed_rotation_angle_deg=ANGLE_DEG, # Ignored
        correction_strategy=strategy_instance, param_name='rotation_angle_deg',
        param_values=list(angle_values_to_test), shape_type=shape_for_param_sweep, **{}
    )
    if results_angle_variation: plot_experiment_results(results_angle_variation, 'rotation_angle_deg')
    else: print("No results generated for angle variation.")
    print("-" * 30)


    # ==================================================================
    # --- 3. Parameter Variation Example: Varying Quantization Level ---
    # ==================================================================
    # (Code unchanged)
    print(f"\n--- Running Parameter Variation (Varying Quant Level, Shape='{shape_for_param_sweep}') ---")
    max_quant_level = max(1, MODULUS // 2) # Allow larger quant levels now
    quantization_values_to_test = list(range(1, max_quant_level + 1))
    results_quant_variation = run_experiment_parameter_variation(
        dim=DIM, modulus=MODULUS, fixed_quantization_level=QUANT_LEVEL, # Ignored
        fixed_rotation_angle_deg=ANGLE_DEG, correction_strategy=strategy_instance,
        param_name='quantization_level', param_values=quantization_values_to_test,
        shape_type=shape_for_param_sweep, **{}
    )
    if results_quant_variation: plot_experiment_results(results_quant_variation, 'quantization_level')
    else: print("No results generated for quantization level variation.")
    print("-" * 30)

    print("\n--- All Experiments Finished ---")
