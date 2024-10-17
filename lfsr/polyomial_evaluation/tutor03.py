import random

class GF2:
    """
    A class representing elements of the Galois Field GF(2).
    """
    def __init__(self, value):
        self.value = value % 2

    def __add__(self, other):
        return GF2(self.value ^ other.value)

    def __mul__(self, other):
        return GF2(self.value & other.value)

    def __eq__(self, other):
        return self.value == other.value

    def __repr__(self):
        return str(self.value)

class Polynomial:
    """
    A class representing polynomials over GF(2).
    """
    def __init__(self, coeffs):
        self.coeffs = [GF2(c) for c in coeffs]

    def __call__(self, x):
        result = GF2(0)
        for i, coeff in enumerate(self.coeffs):
            result += coeff * GF2(x.value ** i)
        return result

    def __repr__(self):
        terms = []
        for i, coeff in enumerate(self.coeffs):
            if coeff.value == 1:
                if i == 0:
                    terms.append("1")
                elif i == 1:
                    terms.append("x")
                else:
                    terms.append(f"x^{i}")
        return " + ".join(terms) or "0"

def polynomial_to_lfsr(poly):
    """
    Convert a polynomial to an LFSR.
    Returns the feedback polynomial and initial state.
    """
    degree = len(poly.coeffs) - 1
    feedback = [GF2(1)] + [GF2(0)] * (degree - 1) + [GF2(1)]
    initial_state = poly.coeffs[:-1]
    return Polynomial(feedback), initial_state

def run_lfsr(feedback, state, steps):
    """
    Run the LFSR for a given number of steps.
    """
    output = []
    for _ in range(steps):
        output.append(state[0])
        new_bit = sum(f * s for f, s in zip(feedback.coeffs, state))
        state = state[1:] + [new_bit]
    return output

# Example usage
def main():
    # Generate a random 5th degree polynomial
    coeffs = [random.randint(0, 1) for _ in range(6)]
    poly = Polynomial(coeffs)
    print(f"Original polynomial: {poly}")

    # Convert to LFSR
    feedback, initial_state = polynomial_to_lfsr(poly)
    print(f"LFSR feedback polynomial: {feedback}")
    print(f"LFSR initial state: {initial_state}")

    # Run LFSR
    steps = 20
    output = run_lfsr(feedback, initial_state, steps)
    print(f"LFSR output (first {steps} bits): {output}")

    # Verify
    print("\nVerification:")
    for i, bit in enumerate(output):
        expected = poly(GF2(i))
        print(f"P({i}) = {expected}, LFSR output: {bit}")
        assert bit == expected, f"Mismatch at step {i}"
    print("All outputs match!")

if __name__ == "__main__":
    main()
