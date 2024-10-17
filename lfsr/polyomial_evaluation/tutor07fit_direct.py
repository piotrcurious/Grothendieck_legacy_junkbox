import numpy as np
from typing import List, Tuple
from sympy import Matrix, GF, symbols, Poly

class GF8Element:
    def __init__(self, value: int):
        self.value = value % 8

    def __add__(self, other: 'GF8Element') -> 'GF8Element':
        return GF8Element(self.value ^ other.value)

    def __mul__(self, other: 'GF8Element') -> 'GF8Element':
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
        return GF8Element(result)

    def __eq__(self, other: 'GF8Element') -> bool:
        return self.value == other.value

    def __repr__(self) -> str:
        return f"GF8({self.value})"

def berlekamp_massey(sequence: List[GF8Element]) -> List[GF8Element]:
    n = len(sequence)
    c = [GF8Element(1)] + [GF8Element(0)] * (n - 1)
    b = [GF8Element(1)] + [GF8Element(0)] * (n - 1)
    l, m, b_len = 0, -1, 1

    for i in range(n):
        d = sequence[i]
        for j in range(1, l + 1):
            d += c[j] * sequence[i - j]
        if d.value == 0:
            m += 1
        elif 2 * l <= i:
            temp = c[:]
            for j in range(b_len):
                c[j + i - m] += d * b[j]
            l, m, b = i + 1 - l, i, temp
            b_len = len(b)
        else:
            for j in range(b_len):
                c[j + i - m] += d * b[j]

    return c[:l + 1]

def fit_lfsr_algebraic_geometry(samples: List[GF8Element], max_degree: int) -> Tuple[List[GF8Element], List[GF8Element]]:
    # Step 1: Use Berlekamp-Massey to find the minimal LFSR
    lfsr_coeffs = berlekamp_massey(samples)
    
    # Step 2: Construct the associated algebraic curve
    x = symbols('x')
    lfsr_poly = Poly(sum(GF(8)(c.value) * x**i for i, c in enumerate(lfsr_coeffs)), x, domain=GF(8))
    
    # Step 3: Find points on the curve
    points = []
    for i in range(8):
        y = lfsr_poly.eval(GF(8)(i))
        points.append((i, int(y)))
    
    # Step 4: Interpolate to find a polynomial of degree <= max_degree
    V = Matrix([[GF(8)(x**i) for i in range(max_degree + 1)] for x, _ in points])
    y = Matrix([GF(8)(y) for _, y in points])
    
    # Solve V * coeffs = y
    coeffs = V.solve(y)
    
    # Convert back to GF8Element
    fitted_coeffs = [GF8Element(int(c)) for c in coeffs]
    
    return lfsr_coeffs, fitted_coeffs

def run_lfsr(coeffs: List[GF8Element], initial_state: List[GF8Element], steps: int) -> List[GF8Element]:
    state = initial_state[:]
    output = []
    for _ in range(steps):
        output.append(state[0])
        feedback = sum((c * s for c, s in zip(coeffs, state)), GF8Element(0))
        state = state[1:] + [feedback]
    return output

def generate_noisy_samples(poly_coeffs: List[GF8Element], num_samples: int, noise_prob: float) -> List[GF8Element]:
    def evaluate_poly(x: GF8Element) -> GF8Element:
        return sum((c * GF8Element(x.value**i) for i, c in enumerate(poly_coeffs)), GF8Element(0))
    
    samples = [evaluate_poly(GF8Element(i % 7 + 1)) for i in range(num_samples)]
    
    # Add noise
    for i in range(len(samples)):
        if np.random.random() < noise_prob:
            samples[i] += GF8Element(np.random.randint(1, 8))
    
    return samples

def main():
    # Generate a random polynomial
    true_poly = [GF8Element(np.random.randint(8)) for _ in range(6)]  # 5th degree polynomial
    print("True polynomial coefficients:", true_poly)

    # Generate noisy samples
    num_samples = 30
    noise_prob = 0.2
    samples = generate_noisy_samples(true_poly, num_samples, noise_prob)

    # Fit LFSR using algebraic geometry methods
    lfsr_coeffs, fitted_poly = fit_lfsr_algebraic_geometry(samples, 5)

    print("LFSR coefficients:", lfsr_coeffs)
    print("Fitted polynomial coefficients:", fitted_poly)

    # Run LFSR
    initial_state = samples[:len(lfsr_coeffs) - 1]
    lfsr_output = run_lfsr(lfsr_coeffs, initial_state, num_samples)

    # Compare LFSR output with original samples
    print("\nComparison:")
    for i, (sample, lfsr) in enumerate(zip(samples, lfsr_output)):
        print(f"Step {i}: Original={sample}, LFSR output={lfsr}")

    # Calculate error
    error = sum(1 for s, l in zip(samples, lfsr_output) if s != l)
    print(f"\nTotal mismatches: {error} out of {num_samples}")

if __name__ == "__main__":
    main()
