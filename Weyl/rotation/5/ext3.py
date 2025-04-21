from sympy import Poly, symbols, GF
from sympy.polys.polytools import div, rem

x = symbols('x')

class PolynomialRingExtensionCorrection(CorrectionStrategy):
    def __init__(self, modulus: int, poly_coeffs: list, alpha: float = 0.01):
        """
        Args:
            modulus (int): Modulus for finite field arithmetic.
            poly_coeffs (list): Coefficients of the irreducible polynomial p(x) for extension ring R[x]/(p(x)).
            alpha (float): Scaling factor for correction.
        """
        self.modulus = modulus
        self.alpha = alpha
        self.p = Poly(poly_coeffs, x, modulus=modulus, domain=GF(modulus))

    def get_correction_matrix(self, initial_matrix: np.ndarray, rotation_angle_deg: float,
                              modular_weyl_x: Operator, modular_weyl_p: Operator) -> np.ndarray:
        dimension = initial_matrix.shape[0]
        q = estimate_quantization_effect_modular(initial_matrix, modular_weyl_x, modular_weyl_p)

        # Generate polynomial ring basis: 1, x, x^2, ..., x^{d-1} mod p(x)
        ring_basis = [Poly(1, x, modulus=self.modulus, domain=GF(self.modulus))]
        for i in range(1, dimension):
            next_poly = (ring_basis[-1] * Poly(x, x, modulus=self.modulus)).trunc(self.modulus)
            next_poly = rem(next_poly, self.p, domain=GF(self.modulus))
            ring_basis.append(next_poly)

        # Build the multiplication matrix by x in R[x]/(p(x))
        mult_matrix = np.zeros((dimension, dimension), dtype=float)
        for i, basis_poly in enumerate(ring_basis):
            prod = rem(basis_poly * Poly(x, x, modulus=self.modulus), self.p)
            for j, coeff in enumerate(prod.all_coeffs()[::-1]):
                if j < dimension:
                    mult_matrix[j, i] = float(coeff)

        correction = np.eye(dimension) + self.alpha * q * mult_matrix
        return correction
