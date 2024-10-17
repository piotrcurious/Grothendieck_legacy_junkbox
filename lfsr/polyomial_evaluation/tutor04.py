import random

class GF8:
    """
    A class representing elements of the Galois Field GF(8) = GF(2^3).
    We represent GF(8) as GF(2)[x]/(x^3 + x + 1).
    """
    def __init__(self, value):
        self.value = value % 8

    def __add__(self, other):
        return GF8(self.value ^ other.value)

    def __mul__(self, other):
        result = 0
        a, b = self.value, other.value
        for _ in range(3):
            if b & 1:
                result ^= a
            high_bit = a & 4
            a <<= 1
            if high_bit:
                a ^= 0b1011  # x^3 + x + 1
            b >>= 1
        return GF8(result)

    def __eq__(self, other):
        return self.value == other.value

    def __repr__(self):
        return f"GF8({self.value})"

class Polynomial:
    """
    A class representing polynomials over GF(8).
    """
    def __init__(self, coeffs):
        self.coeffs = [GF8(c) for c in coeffs]

    def __call__(self, x):
        result = GF8(0)
        for i, coeff in enumerate(self.coeffs):
            result += coeff * x**i
        return result

    def __repr__(self):
        terms = []
        for i, coeff in enumerate(self.coeffs):
            if coeff.value != 0:
                if i == 0:
                    terms.append(f"{coeff}")
                elif i == 1:
                    terms.append(f"{coeff}x")
                else:
                    terms.append(f"{coeff}x^{i}")
        return " + ".join(terms) or "0"

def polynomial_to_lfsr(poly):
    """
    Convert a polynomial to an LFSR over GF(8).
    Returns the feedback polynomial and initial state.
    """
    degree = len(poly.coeffs) - 1
    feedback = [GF8(1)] + [GF8(0)] * (degree - 1) + [poly.coeffs[-1]]
    initial_state = poly.coeffs[:-1]
    return Polynomial(feedback), initial_state

def run_lfsr(feedback, state, steps):
    """
    Run the LFSR for a given number of steps.
    """
    output = []
    for _ in range(steps):
        output.append(state[0])
        new_element = sum((f * s for f, s in zip(feedback.coeffs, state)), GF8(0))
        state = state[1:] + [new_element]
    return output

def main():
    # Generate a random 5th degree polynomial over GF(8)
    coeffs = [random.randint(0, 7) for _ in range(6)]
    poly = Polynomial(coeffs)
    print(f"Original polynomial: {poly}")

    # Convert to LFSR
    feedback, initial_state = polynomial_to_lfsr(poly)
    print(f"LFSR feedback polynomial: {feedback}")
    print(f"LFSR initial state: {initial_state}")

    # Run LFSR
    steps = 20
    output = run_lfsr(feedback, initial_state, steps)
    print(f"LFSR output (first {steps} elements): {output}")

    # Verify
    print("\nVerification:")
    x = GF8(2)  # We use x = 2 as a generator of GF(8)*
    x_power = GF8(1)
    for i, element in enumerate(output):
        expected = poly(x_power)
        print(f"P({x_power}) = {expected}, LFSR output: {element}")
        assert element == expected, f"Mismatch at step {i}"
        x_power *= x
    print("All outputs match!")

if __name__ == "__main__":
    main()
