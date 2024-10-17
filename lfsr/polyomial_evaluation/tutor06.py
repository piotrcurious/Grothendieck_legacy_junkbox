import numpy as np
from typing import List, Tuple

class GF8:
    def __init__(self, value: int):
        self.value = value % 8

    def __add__(self, other: 'GF8') -> 'GF8':
        return GF8(self.value ^ other.value)

    def __sub__(self, other: 'GF8') -> 'GF8':
        return self + other  # In GF(8), addition and subtraction are the same

    def __mul__(self, other: 'GF8') -> 'GF8':
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

    def __truediv__(self, other: 'GF8') -> 'GF8':
        for i in range(8):
            if self == other * GF8(i):
                return GF8(i)
        raise ValueError("Division by zero")

    def __eq__(self, other: 'GF8') -> bool:
        return self.value == other.value

    def __repr__(self) -> str:
        return f"GF8({self.value})"

class Polynomial:
    def __init__(self, coeffs: List[GF8]):
        self.coeffs = coeffs

    def __call__(self, x: GF8) -> GF8:
        result = GF8(0)
        for i, coeff in enumerate(self.coeffs):
            result += coeff * GF8(x.value ** i)
        return result

    def __repr__(self) -> str:
        return " + ".join([f"{coeff}x^{i}" for i, coeff in enumerate(self.coeffs) if coeff.value != 0]) or "0"

def polynomial_to_lfsr(poly: Polynomial) -> Tuple[Polynomial, List[GF8]]:
    degree = len(poly.coeffs) - 1
    feedback = [GF8(1)] + [GF8(0)] * (degree - 1) + [poly.coeffs[-1]]
    initial_state = poly.coeffs[:-1]
    return Polynomial(feedback), initial_state

def run_lfsr(feedback: Polynomial, state: List[GF8], steps: int) -> List[GF8]:
    output = []
    for _ in range(steps):
        output.append(state[0])
        new_element = sum((f * s for f, s in zip(feedback.coeffs, state)), GF8(0))
        state = state[1:] + [new_element]
    return output

def gf8_to_numpy(value: GF8) -> np.ndarray:
    return np.array([int(bool(value.value & (1 << i))) for i in range(3)])

def numpy_to_gf8(array: np.ndarray) -> GF8:
    return GF8(sum(int(bit) << i for i, bit in enumerate(array)))

def least_squares_fit(x: List[GF8], y: List[GF8], degree: int) -> Polynomial:
    X = np.array([gf8_to_numpy(GF8(x_i.value ** j)) for x_i in x for j in range(degree + 1)]).reshape(-len(x), degree + 1)
    Y = np.array([gf8_to_numpy(y_i) for y_i in y]).reshape(-1, 3)
    
    XTX = X.T @ X
    XTY = X.T @ Y
    
    coeffs = np.linalg.lstsq(XTX, XTY, rcond=None)[0]
    
    return Polynomial([numpy_to_gf8(coeff) for coeff in coeffs])

def generate_noisy_samples(poly: Polynomial, num_samples: int, noise_prob: float) -> Tuple[List[GF8], List[GF8]]:
    x_values = [GF8(i % 7 + 1) for i in range(num_samples)]  # Use non-zero elements of GF(8)
    y_values = [poly(x) for x in x_values]
    
    # Add noise
    for i in range(len(y_values)):
        if np.random.random() < noise_prob:
            y_values[i] += GF8(np.random.randint(1, 8))  # Add non-zero noise
    
    return x_values, y_values

def main():
    # Generate a random polynomial
    true_poly = Polynomial([GF8(np.random.randint(8)) for _ in range(6)])  # 5th degree polynomial
    print(f"True polynomial: {true_poly}")

    # Generate noisy samples
    num_samples = 30
    noise_prob = 0.2
    x_samples, y_samples = generate_noisy_samples(true_poly, num_samples, noise_prob)

    # Fit polynomial using least squares
    fitted_poly = least_squares_fit(x_samples, y_samples, 5)
    print(f"Fitted polynomial: {fitted_poly}")

    # Convert fitted polynomial to LFSR
    feedback, initial_state = polynomial_to_lfsr(fitted_poly)
    print(f"LFSR feedback polynomial: {feedback}")
    print(f"LFSR initial state: {initial_state}")

    # Generate sequence using LFSR
    lfsr_output = run_lfsr(feedback, initial_state, num_samples)

    # Compare LFSR output with original samples
    print("\nComparison:")
    for i, (x, y, lfsr) in enumerate(zip(x_samples, y_samples, lfsr_output)):
        print(f"Step {i}: x={x}, Original y={y}, LFSR output={lfsr}")

    # Calculate error
    error = sum(1 for y, lfsr in zip(y_samples, lfsr_output) if y != lfsr)
    print(f"\nTotal mismatches: {error} out of {num_samples}")

if __name__ == "__main__":
    main()
