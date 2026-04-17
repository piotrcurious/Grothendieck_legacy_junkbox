from sympy import Matrix, symbols, Poly, GF, simplify
from sympy.polys.polytools import rem

x = symbols('x')

class PolynomialMatrixCorrectionStrategy(CorrectionStrategy):
    def __init__(self, modulus: int, poly_mod_coeffs: list, degree: int = 1):
        """
        Args:
            modulus (int): Field characteristic.
            poly_mod_coeffs (list): Coefficients of irreducible polynomial p(x) to define quotient ring.
            degree (int): Degree of polynomial matrix correction.
        """
        self.modulus = modulus
        self.degree = degree
        self.px = Poly(poly_mod_coeffs, x, domain=GF(modulus))

    def _make_polynomial_matrix(self, q: float, dim: int) -> Matrix:
        # Initialize polynomial matrix: M(x) = A0 + A1*x + A2*x^2 + ...
        poly_matrix = Matrix.zeros(dim)

        coeffs = []
        for d in range(self.degree + 1):
            # Construct each Ai matrix coefficient
            Ai = Matrix.eye(dim) * (q ** (d + 1)) * (1 / (d + 1))
            coeffs.append(Ai)

        # Build matrix-valued polynomial: M(x) = A0 + A1*x + A2*x^2 ...
        poly_matrix_expr = sum([Ai * x**i for i, Ai in enumerate(coeffs)])
        # Reduce modulo p(x)
        return rem(poly_matrix_expr, self.px)

    def get_correction_matrix(self, initial_matrix: np.ndarray, rotation_angle_deg: float,
                              modular_weyl_x: Operator, modular_weyl_p: Operator) -> Poly:
        dim = initial_matrix.shape[0]
        q = estimate_quantization_effect_modular(initial_matrix, modular_weyl_x, modular_weyl_p)

        # Construct symbolic polynomial matrix
        poly_matrix = self._make_polynomial_matrix(q, dim)
        return simplify(poly_matrix)
