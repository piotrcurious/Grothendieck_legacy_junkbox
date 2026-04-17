import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Type, Tuple

# Enhanced function for quantization effect estimation
def enhanced_estimate_quantization_effect(matrix: NDArray, weyl_x: Operator, weyl_p: Operator, 
                                         angle_deg: float) -> Dict[str, float]:
    """
    Enhanced estimation of quantization effects that accounts for rotation angle.
    Returns both the commutator norm and an angle-dependent weighting factor.
    
    Args:
        matrix: Input matrix
        weyl_x: X (position shift) operator
        weyl_p: P (momentum shift) operator
        angle_deg: Rotation angle in degrees
        
    Returns:
        Dictionary with commutator norm, angle factor, and effective Q
    """
    px_matrix = weyl_x(matrix)
    p_of_x = weyl_p(px_matrix)
    xp_matrix = weyl_p(matrix)
    x_of_p = weyl_x(xp_matrix)
    commutator = p_of_x - x_of_p
    norm_q = float(np.linalg.norm(commutator, 'fro'))
    
    # Angle dependency - non-commutativity matters more at certain angles
    # Most critical at 45°, 135°, 225°, 315° (i.e., not aligned with axes)
    angle_rad = np.radians(angle_deg % 90)
    angle_factor = np.sin(2 * angle_rad)  # Peak at 45°, zero at 0° and 90°
    
    # Weighted quantization effect
    effective_q = norm_q * (1 + angle_factor)
    
    return {
        'commutator_norm': norm_q,
        'angle_factor': angle_factor,
        'effective_q': effective_q
    }

# Enhanced Lemma-Based Correction Strategy
class EnhancedLemmaBasedCorrectionStrategy(CorrectionStrategy):
    """
    Enhanced correction strategy using field extension principles that accounts for 
    both the quantization effect and the rotation angle.
    
    This implementation improves the mathematical foundation by:
    1. Incorporating angle-dependent correction that respects field extension properties
    2. Properly accounting for the non-commutativity in the Weyl algebra
    3. Providing diagnostic information for analysis
    """
    def __init__(self, correction_factor: float = 0.01, angle_sensitivity: float = 0.5) -> None:
        """
        Initialize the enhanced lemma-based correction strategy.
        
        Args:
            correction_factor: Controls the base strength of the correction
            angle_sensitivity: Controls how strongly rotation angle affects correction
        """
        if correction_factor < 0:
            print(f"Warning: Negative correction_factor ({correction_factor}) is unusual.")
        self.correction_factor = correction_factor
        self.angle_sensitivity = angle_sensitivity  # Controls how angle affects correction
        self._last_q = None
        self._last_scale = None
        self._last_angle_factor = None
        self._last_effective_q = None
        
    def get_correction_matrix(self, matrix: NDArray, angle_deg: float, weyl_x: Operator, weyl_p: Operator) -> NDArray:
        """
        Calculates correction using both Q and angle information.
        Uses field extension principles to account for non-commutative structure.
        
        Args:
            matrix: Input matrix to be corrected
            angle_deg: Rotation angle in degrees
            weyl_x: X (position shift) operator
            weyl_p: P (momentum shift) operator
            
        Returns:
            Correction matrix to be applied to the input matrix
        """
        # Get enhanced quantization effect metrics
        q_metrics = enhanced_estimate_quantization_effect(matrix, weyl_x, weyl_p, angle_deg)
        Q = q_metrics['commutator_norm']
        angle_factor = q_metrics['angle_factor']
        effective_q = q_metrics['effective_q']
        
        # Combined scaling factor considering both Q and angle
        correction_strength = self.correction_factor * Q * (1 + self.angle_sensitivity * angle_factor)
        scale = 1.0 / (1.0 + correction_strength)
        
        # Store for diagnostics
        self._last_q = Q
        self._last_angle_factor = angle_factor
        self._last_effective_q = effective_q
        self._last_scale = scale
        
        return np.eye(matrix.shape[0], dtype=float) * scale
    
    # Helper methods for diagnostics
    def get_last_q_value(self) -> Optional[float]:
        return self._last_q
    
    def get_last_scale_factor(self) -> Optional[float]:
        return self._last_scale
    
    def get_last_angle_factor(self) -> Optional[float]:
        return self._last_angle_factor
    
    def get_last_effective_q(self) -> Optional[float]:
        return self._last_effective_q
    
    def get_diagnostics(self) -> Dict[str, Optional[float]]:
        """Returns all diagnostic values from the last correction calculation."""
        return {
            'commutator_norm': self._last_q,
            'angle_factor': self._last_angle_factor,
            'effective_q': self._last_effective_q,
            'scale_factor': self._last_scale
        }


# Weyl Algebra-Based Correction Strategy
class WeylAlgebraBasedCorrectionStrategy(CorrectionStrategy):
    """
    Advanced correction strategy based on the structure of the Weyl algebra.
    
    Uses the commutator [X,P] to construct a non-uniform correction matrix that 
    accounts for the non-commutativity of the phase space. This strategy better 
    respects the mathematical structure of the Weyl algebra field extension.
    """
    def __init__(self, correction_factor: float = 0.01, commutator_weight: float = 0.2) -> None:
        """
        Initialize the Weyl algebra-based correction strategy.
        
        Args:
            correction_factor: Controls the base strength of the correction
            commutator_weight: Controls how strongly the commutator structure affects the correction
        """
        self.correction_factor = correction_factor
        self.commutator_weight = commutator_weight
        self._last_q = None
        self._last_scale = None
        self._last_commutator_contribution = None
        
    def get_correction_matrix(self, matrix: NDArray, angle_deg: float, weyl_x: Operator, weyl_p: Operator) -> NDArray:
        """
        Constructs a correction matrix using the Weyl algebra structure.
        
        Args:
            matrix: Input matrix to be corrected
            angle_deg: Rotation angle in degrees
            weyl_x: X (position shift) operator
            weyl_p: P (momentum shift) operator
            
        Returns:
            Non-uniform correction matrix derived from Weyl algebra structure
        """
        dim = matrix.shape[0]
        
        # Calculate commutator [X,P] applied to the matrix
        px_matrix = weyl_x(matrix)
        p_of_x = weyl_p(px_matrix)
        xp_matrix = weyl_p(matrix)
        x_of_p = weyl_x(xp_matrix)
        commutator = p_of_x - x_of_p
        
        # Normalize the commutator
        Q = float(np.linalg.norm(commutator, 'fro'))
        self._last_q = Q
        
        # Angle factor (more correction needed at angles not aligned with axes)
        angle_rad = np.radians(angle_deg % 90)
        angle_factor = np.sin(2 * angle_rad)  # Peak at 45°, zero at 0° and 90°
        
        # Basic scaling as in the original strategy but with angle consideration
        basic_scale = 1.0 / (1.0 + self.correction_factor * Q * (1 + angle_factor))
        self._last_scale = basic_scale
        
        # Create identity matrix with basic scaling
        correction = np.eye(dim, dtype=float) * basic_scale
        
        # Add a perturbation based on the commutator structure
        # This directly incorporates the non-commutative structure of the Weyl algebra
        commutator_contribution = 0.0
        if Q > 0:
            # Normalize and scale the commutator based on the angle and Q
            # The contribution is larger when non-commutativity is significant
            norm_commutator = commutator / Q
            contribution_scale = (1 - basic_scale) * self.commutator_weight * angle_factor
            commutator_contribution = contribution_scale
            
            # Add the contribution, ensuring it's appropriately scaled
            correction += norm_commutator * contribution_scale
        
        self._last_commutator_contribution = commutator_contribution
        return correction
    
    # Helper methods for diagnostics
    def get_last_q_value(self) -> Optional[float]:
        return self._last_q
    
    def get_last_scale_factor(self) -> Optional[float]:
        return self._last_scale
    
    def get_last_commutator_contribution(self) -> Optional[float]:
        return self._last_commutator_contribution


# Field Extension Matrix Factory - High-level interface for creating field extension matrices
class FieldExtensionMatrixFactory:
    """
    Factory class for creating field extension matrices based on Weyl algebra structure.
    
    This class provides a high-level interface for generating correction matrices
    that respect the mathematical structures of modular phase spaces and field extensions.
    """
    @staticmethod
    def create_basic_extension(dim: int, scale: float) -> NDArray:
        """Creates a basic scalar field extension matrix."""
        return np.eye(dim, dtype=float) * scale
    
    @staticmethod
    def create_from_commutator(commutator: NDArray, base_scale: float, weight: float) -> NDArray:
        """Creates a field extension matrix incorporating commutator structure."""
        dim = commutator.shape[0]
        # Normalize commutator
        norm = np.linalg.norm(commutator, 'fro')
        if norm > 0:
            norm_commutator = commutator / norm
            # Base scaling plus commutator contribution
            return np.eye(dim, dtype=float) * base_scale + norm_commutator * weight
        else:
            # If commutator is zero, just return scaled identity
            return np.eye(dim, dtype=float) * base_scale
    
    @staticmethod
    def analyze_field_extension(correction_matrix: NDArray) -> Dict[str, float]:
        """
        Analyzes properties of a field extension matrix.
        
        Returns:
            Dictionary with metrics like diagonal dominance, symmetry, etc.
        """
        # Extract metrics from the correction matrix
        eigenvalues = np.linalg.eigvals(correction_matrix)
        off_diag_norm = np.linalg.norm(correction_matrix - np.diag(np.diag(correction_matrix)), 'fro')
        diag_norm = np.linalg.norm(np.diag(np.diag(correction_matrix)), 'fro')
        
        return {
            'min_eigenvalue': float(np.min(np.real(eigenvalues))),
            'max_eigenvalue': float(np.max(np.real(eigenvalues))),
            'diagonal_dominance': float(diag_norm / (off_diag_norm + 1e-10)),
            'off_diagonal_contribution': float(off_diag_norm / (np.linalg.norm(correction_matrix, 'fro') + 1e-10))
        }


# Helper functions for analyzing correction performance
def analyze_correction_effectiveness(original: NDArray, standard_rotated: NDArray, 
                                    corrected_rotated: NDArray, correction_matrix: NDArray,
                                    weyl_x: Operator, weyl_p: Operator, angle_deg: float) -> Dict[str, float]:
    """
    Comprehensive analysis of correction effectiveness.
    
    Args:
        original: Original matrix
        standard_rotated: Matrix after standard rotation
        corrected_rotated: Matrix after corrected rotation
        correction_matrix: Applied correction matrix
        weyl_x: X operator
        weyl_p: P operator
        angle_deg: Rotation angle in degrees
        
    Returns:
        Dictionary with various quality and diagnostic metrics
    """
    # Standard metrics
    overlap_std = float(np.sum(original * standard_rotated))
    overlap_corr = float(np.sum(original * corrected_rotated))
    diff_norm = float(np.linalg.norm(corrected_rotated - standard_rotated, 'fro'))
    
    # Field extension analysis
    extension_metrics = FieldExtensionMatrixFactory.analyze_field_extension(correction_matrix)
    
    # Quantization effect
    q_metrics = enhanced_estimate_quantization_effect(original, weyl_x, weyl_p, angle_deg)
    
    # Combine all metrics
    metrics = {
        'overlap_standard': overlap_std,
        'overlap_corrected': overlap_corr,
        'difference_norm': diff_norm,
        'improvement_ratio': float(overlap_corr / (overlap_std + 1e-10)),
    }
    
    # Add field extension metrics
    metrics.update({f"extension_{k}": v for k, v in extension_metrics.items()})
    
    # Add quantization metrics
    metrics.update({f"quantization_{k}": v for k, v in q_metrics.items()})
    
    return metrics


# Enhanced visualization functions
def display_correction_matrix_structure(correction_matrix: NDArray, title: str) -> plt.Figure:
    """
    Visualizes the structure of a correction matrix to inspect field extension properties.
    
    Args:
        correction_matrix: Correction matrix to visualize
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot correction matrix
    vmax = max(np.abs(correction_matrix).max(), 1e-6)
    im0 = axes[0].imshow(correction_matrix, cmap='coolwarm', interpolation='nearest', 
                         vmin=-vmax, vmax=vmax)
    axes[0].set_title("Correction Matrix Structure")
    axes[0].axis('on')
    axes[0].set_aspect('equal')
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Plot deviation from identity
    deviation = correction_matrix - np.eye(correction_matrix.shape[0]) * correction_matrix[0,0]
    vmax_dev = max(np.abs(deviation).max(), 1e-6)
    im1 = axes[1].imshow(deviation, cmap='coolwarm', interpolation='nearest',
                        vmin=-vmax_dev, vmax=vmax_dev)
    axes[1].set_title(f"Deviation from Uniform Scaling\nMax Dev: {vmax_dev:.6f}")
    axes[1].axis('on')
    axes[1].set_aspect('equal')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Add analysis as text
    metrics = FieldExtensionMatrixFactory.analyze_field_extension(correction_matrix)
    metrics_str = "\n".join([f"{k}: {v:.6f}" for k, v in metrics.items()])
    fig.text(0.02, 0.02, f"Analysis:\n{metrics_str}", fontsize=10, 
             verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    return fig


def plot_enhanced_experiment_results(results: List[Dict[str, Any]], param_name: str) -> plt.Figure:
    """
    Enhanced plotting of experiment results with additional metrics.
    
    Args:
        results: List of result dictionaries from experiments
        param_name: Parameter that was varied in the experiment
        
    Returns:
        Matplotlib figure
    """
    if not results:
        raise ValueError("Results list is empty.")
        
    first_result = results[0]
    modulus = first_result['modulus']
    strategy_name = first_result['strategy']
    shape_type = first_result['shape_type']
    
    fixed_param_info = ""
    if param_name == 'rotation_angle_deg':
        fixed_param_info = f"Quant Level = {first_result['quantization_level']}"
    elif param_name == 'quantization_level':
        fixed_param_info = f"Angle = {first_result['rotation_angle_deg']:.1f}°"
    
    # Extract data series
    param_vals = [r[param_name] for r in results]
    overlap_std = [r['overlap_standard'] for r in results]
    overlap_corr = [r['overlap_corrected'] for r in results]
    diff_norm = [r['difference_norm'] for r in results]
    
    # Extract new metrics if available
    has_improved_metrics = 'improvement_ratio' in results[0]
    if has_improved_metrics:
        improvement_ratio = [r['improvement_ratio'] for r in results]
        quant_effect = [r.get('quantization_effective_q', r.get('quantization_commutator_norm', 0)) for r in results]
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(14, 10))
    if has_improved_metrics:
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
    else:
        gs = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
    
    # Plot 1: Overlap metrics
    color1 = 'tab:blue'
    ax1.set_xlabel(f'{param_name.replace("_", " ").title()}')
    ax1.set_ylabel('Overlap with Original', color=color1)
    line1, = ax1.plot(param_vals, overlap_std, 'o-', color=color1, label='Standard Rotation Overlap')
    line2, = ax1.plot(param_vals, overlap_corr, 's-', color='tab:cyan', label='Corrected Rotation Overlap')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, axis='y', linestyle=':', linewidth=0.5)
    ax1.set_title("Overlap Metrics")
    lines1 = [line1, line2]
    labels1 = [l.get_label() for l in lines1]
    ax1.legend(lines1, labels1, loc='best')
    
    # Plot 2: Difference norm
    color2 = 'tab:red'
    ax2.set_xlabel(f'{param_name.replace("_", " ").title()}')
    ax2.set_ylabel('Norm(Corrected - Standard)', color=color2)
    line3, = ax2.plot(param_vals, diff_norm, '^--', color=color2, label='Difference Norm')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.grid(True, axis='y', linestyle=':', linewidth=0.5)
    ax2.set_title("Difference Norm")
    ax2.legend(loc='best')
    
    # Additional plots for enhanced metrics
    if has_improved_metrics:
        # Plot 3: Improvement ratio
        ax3.set_xlabel(f'{param_name.replace("_", " ").title()}')
        ax3.set_ylabel('Improvement Ratio')
        line4, = ax3.plot(param_vals, improvement_ratio, 'D-', color='tab:green', label='Improvement Ratio')
        ax3.tick_params(axis='y')
        ax3.grid(True, axis='y', linestyle=':', linewidth=0.5)
        ax3.set_title("Improvement Ratio (Higher is Better)")
        ax3.legend(loc='best')
        
        # Plot 4: Quantization effect
        ax4.set_xlabel(f'{param_name.replace("_", " ").title()}')
        ax4.set_ylabel('Quantization Effect')
        line5, = ax4.plot(param_vals, quant_effect, '*-', color='tab:purple', label='Quantization Effect')
        ax4.tick_params(axis='y')
        ax4.grid(True, axis='y', linestyle=':', linewidth=0.5)
        ax4.set_title("Quantization Effect Metric")
        ax4.legend(loc='best')
    
    plt.suptitle(f'Rotation Quality vs {param_name.replace("_", " ").title()}\n' + 
                 f'(Shape: {shape_type}, Mod: {modulus}, {fixed_param_info}, Strategy: {strategy_name})', 
                 fontsize=14)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


# Function to run a comparative experiment with multiple strategies
def run_comparative_experiment(dim: int, modulus: int, quant_level: int, angle_deg: float,
                              shape_type: str, strategies: List[CorrectionStrategy]) -> Dict[str, Any]:
    """
    Runs a comprehensive experiment comparing multiple correction strategies.
    
    Args:
        dim: Matrix dimension
        modulus: Phase space modulus
        quant_level: Quantization level
        angle_deg: Rotation angle in degrees
        shape_type: Shape to use for testing ('circle', 'square', etc.)
        strategies: List of correction strategies to compare
        
    Returns:
        Dictionary with results for each strategy
    """
    phase_space = ModularPhaseSpace(dim, modulus)
    weyl_x = ModularWeylXOperator(phase_space, quant_level)
    weyl_p = ModularWeylPOperator(phase_space, quant_level)
    initial_matrix = create_shape_matrix(shape_type, dim)
    standard_rotated = rotate_matrix_around_center(initial_matrix, angle_deg)
    
    results = {}
    for strategy in strategies:
        strategy_name = str(strategy)
        print(f"Testing strategy: {strategy_name}")
        
        # Apply correction
        correction_matrix = strategy.get_correction_matrix(initial_matrix, angle_deg, weyl_x, weyl_p)
        corrected_rotated = apply_corrected_rotation(initial_matrix, correction_matrix, angle_deg)
        
        # Analyze results
        metrics = analyze_correction_effectiveness(
            original=initial_matrix,
            standard_rotated=standard_rotated,
            corrected_rotated=corrected_rotated,
            correction_matrix=correction_matrix,
            weyl_x=weyl_x,
            weyl_p=weyl_p,
            angle_deg=angle_deg
        )
        
        # Store results
        results[strategy_name] = {
            'metrics': metrics,
            'correction_matrix': correction_matrix,
            'corrected_result': corrected_rotated
        }
        
        # Print key metrics
        print(f"  Overlap with original: {metrics['overlap_corrected']:.4f}")
        print(f"  Improvement over standard: {metrics['improvement_ratio']:.4f}")
        
    return {
        'initial_matrix': initial_matrix,
        'standard_rotated': standard_rotated,
        'strategy_results': results,
        'parameters': {
            'dimension': dim,
            'modulus': modulus,
            'quant_level': quant_level,
            'angle_deg': angle_deg,
            'shape_type': shape_type
        }
    }


# Helper function to visualize comparative results
def display_comparative_results(experiment_results: Dict[str, Any]) -> None:
    """
    Creates visualizations comparing different correction strategies.
    
    Args:
        experiment_results: Results from run_comparative_experiment
    """
    initial_matrix = experiment_results['initial_matrix']
    standard_rotated = experiment_results['standard_rotated']
    strategy_results = experiment_results['strategy_results']
    params = experiment_results['parameters']
    
    # Create overall comparison figure
    num_strategies = len(strategy_results)
    fig, axes = plt.subplots(1, num_strategies + 2, figsize=(5 * (num_strategies + 2), 6))
    
    # Display original
    vmax = max(initial_matrix.max(), standard_rotated.max(), 
               max(r['corrected_result'].max() for r in strategy_results.values()), 1e-6)
    vmin = 0
    
    axes[0].imshow(initial_matrix, cmap='gray', interpolation='nearest', vmin=vmin, vmax=vmax)
    axes[0].set_title("Original Shape")
    axes[0].axis('off')
    axes[0].set_aspect('equal')
    
    # Display standard rotation
    axes[1].imshow(standard_rotated, cmap='gray', interpolation='nearest', vmin=vmin, vmax=vmax)
    axes[1].set_title("Standard Rotation")
    axes[1].axis('off')
    axes[1].set_aspect('equal')
    
    # Display each strategy result
    for i, (strategy_name, result) in enumerate(strategy_results.items()):
        corrected = result['corrected_result']
        metrics = result['metrics']
        
        axes[i+2].imshow(corrected, cmap='gray', interpolation='nearest', vmin=vmin, vmax=vmax)
        axes[i+2].set_title(f"{strategy_name}\nOverlap: {metrics['overlap_corrected']:.3f}\nImprovement: {metrics['improvement_ratio']:.3f}x")
        axes[i+2].axis('off')
        axes[i+2].set_aspect('equal')
    
    plt.suptitle(f"Comparison of Correction Strategies\n" +
                f"Shape: {params['shape_type']}, Dim: {params['dimension']}, " +
                f"Mod: {params['modulus']}, Quant: {params['quant_level']}, " +
                f"Angle: {params['angle_deg']}°", 
                fontsize=14)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
    # For each strategy, show its correction matrix structure
    for strategy_name, result in strategy_results.items():
        correction_matrix = result['correction_matrix']
        fig = display_correction_matrix_structure(
            correction_matrix, 
            f"Correction Matrix Structure - {strategy_name}"
        )
        plt.show()


# SAMPLE USAGE EXAMPLE

if __name__ == "__main__":
    print("Enhanced Weyl Algebra and Field Extension Corrections")
    print("-" * 60)
    
    # Configuration
    DIM = 64
    MODULUS = 64
    QUANT_LEVEL = 5
    ANGLE_DEG = 45.0  # 45° is where non-commutativity has maximum effect
    SHAPE_TYPE = 'circle'
    
    # Create different correction strategies to compare
    strategies = [
        LemmaBasedCorrectionStrategy(correction_factor=0.01),  # Original
        EnhancedLemmaBasedCorrectionStrategy(correction_factor=0.01, angle_sensitivity=0.5),  # Enhanced
        WeylAlgebraBasedCorrectionStrategy(correction_factor=0.01, commutator_weight=0.2),  # Full Weyl algebra
    ]
    
    # Run comparative experiment
    results = run_comparative_experiment(
        dim=DIM, 
        modulus=MODULUS, 
        quant_level=QUANT_LEVEL,
        angle_deg=ANGLE_DEG,
        shape_type=SHAPE_TYPE,
        strategies=strategies
    )
    
    # Display comparative results
    display_comparative_results(results)
    
    print("-" * 60)
    print("Experiment complete!")
