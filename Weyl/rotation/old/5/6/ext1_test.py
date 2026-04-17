import numpy as np from sympy import symbols, Poly, GF

Assuming all classes and functions are defined/imported in this module:

ModularPhaseSpace, ModularWeylXOperator, ModularWeylPOperator,

discrete_rotation_operator, apply_modular_corrected_rotation,

LemmaBasedCorrectionStrategy, RandomNoiseCorrectionStrategy,

PolynomialMatrixCorrectionStrategy, create_shape_matrix,

calculate_rotation_quality

x = symbols('x')

def run_rotation_tests( dimension: int = 16, modulus: int = 7, poly_mod_coeffs: list = [1, 0, 1, 1],  # default x^3 + x + 1 shapes: list = None, angles: list = None, strategies: dict = None ) -> None: """ Runs rotation tests comparing standard vs. corrected rotations for various strategies. Prints quality metrics for each combination. """ # Defaults if shapes is None: shapes = ['square', 'circle', 'triangle', 'cross'] if angles is None: angles = [0, 15, 30, 45, 60, 90]

# Initialize phase space and Weyl operators
phase_space = ModularPhaseSpace(dimension, modulus)
weyl_x = ModularWeylXOperator(phase_space)
weyl_p = ModularWeylPOperator(phase_space)

# Define correction strategies
if strategies is None:
    strategies = {
        'none': None,
        'lemma': LemmaBasedCorrectionStrategy(correction_factor=0.005),
        'noise': RandomNoiseCorrectionStrategy(noise_factor=0.01),
        'poly_ext': PolynomialMatrixCorrectionStrategy(modulus, poly_mod_coeffs, degree=2)
    }

print(f"Rotation Test Suite: dimension={dimension}, modulus={modulus}\n")

# Table header
header = ["Shape", "Angle", "Strategy", "Overlap Std", "Overlap Corr", "Diff Norm"]
print("\t".join(header))

for shape in shapes:
    original = create_shape_matrix(shape, dimension)
    for angle in angles:
        rot_mat = discrete_rotation_operator(angle, dimension)
        rotated_std = rot_mat @ original @ rot_mat.T

        for name, strat in strategies.items():
            if strat is None:
                rotated_corr = rotated_std
            else:
                corr_mat = strat.get_correction_matrix(original, angle, weyl_x, weyl_p)
                # For symbolic polynomial matrices, evaluate at x=1 for numeric test
                if hasattr(corr_mat, 'subs'):
                    corr_mat = np.array(
                        [[float(corr_mat[i, j].subs(x, 1) % modulus) for j in range(dimension)]
                         for i in range(dimension)]
                    )
                rotated_corr = apply_modular_corrected_rotation(original, rot_mat, corr_mat)

            metrics = calculate_rotation_quality(
                original, rotated_std, rotated_corr
            )

            print(f"{shape}\t{angle}\t{name}\t"
                  f"{metrics['overlap_standard']:.2f}\t"
                  f"{metrics['overlap_corrected']:.2f}\t"
                  f"{metrics['difference_norm']:.2f}")

if name == 'main': run_rotation_tests()

