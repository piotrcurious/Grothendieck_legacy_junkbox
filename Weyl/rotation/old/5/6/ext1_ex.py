modulus = 7
irreducible_poly = [1, 0, 1, 1]  # x^3 + x + 1
strategy = PolynomialMatrixCorrectionStrategy(modulus, irreducible_poly, degree=2)

symbolic_correction = strategy.get_correction_matrix(shape, angle, weyl_x, weyl_p)
print("Polynomial correction matrix:\n", symbolic_correction)
