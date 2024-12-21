class PolynomialF2:
    def __init__(self, coeffs):
        """Initialize a polynomial over F2."""
        self.coeffs = [c % 2 for c in coeffs]  # Ensure coefficients are mod 2

    def __repr__(self):
        terms = [f"x^{i}" if c else "" for i, c in enumerate(self.coeffs)]
        return " + ".join(filter(None, terms[::-1])) or "0"

    def __add__(self, other):
        """Addition as vector space operation over F2."""
        max_len = max(len(self.coeffs), len(other.coeffs))
        result = [(self.coeffs[i] if i < len(self.coeffs) else 0) ^
                  (other.coeffs[i] if i < len(other.coeffs) else 0)
                  for i in range(max_len)]
        return PolynomialF2(result)

    def __mul__(self, other):
        """Multiplication as morphism."""
        result = [0] * (len(self.coeffs) + len(other.coeffs) - 1)
        for i, a in enumerate(self.coeffs):
            for j, b in enumerate(other.coeffs):
                result[i + j] ^= a & b
        return PolynomialF2(result)

  class IntegerScheme:
    def __init__(self, value, bits=32):
        """Represent an integer as a polynomial over F2."""
        self.bits = bits
        self.poly = PolynomialF2([int(b) for b in bin(value & ((1 << bits) - 1))[2:].zfill(bits)])

    def add(self, other):
        """Addition in the scheme."""
        return IntegerScheme(int(str(self.poly + other.poly), 2), self.bits)

    def extract_features(poly):
    """Extract features from a polynomial."""
    return {
        "degree": len(poly.coeffs) - 1,
        "leading_coefficient": poly.coeffs[-1] if poly.coeffs else 0,
        "num_terms": sum(poly.coeffs)
    }

def fit_polynomial(data, degree):
    """Fit a polynomial of specified degree to data."""
    from numpy.polynomial.polynomial import Polynomial
    x, y = zip(*data)
    poly = Polynomial.fit(x, y, degree)
    return PolynomialF2([int(c) for c in poly.coef % 2])

class TimeSeries:
    def __init__(self, data):
        """Initialize with timestamp/value pairs."""
        self.data = data

    def extract_features(self):
        """Extract features from the time series."""
        return [extract_features(IntegerScheme(value).poly) for _, value in self.data]

  # Represent an integer
int1 = IntegerScheme(42)
int2 = IntegerScheme(15)
print("Addition:", int1.add(int2).poly)

# Feature extraction
features = extract_features(int1.poly)
print("Features:", features)

# Fit a polynomial
time_series = [(0, 42), (1, 15), (2, 36)]
fitted_poly = fit_polynomial(time_series, degree=2)
print("Fitted Polynomial:", fitted_poly)

# Time series features
ts = TimeSeries(time_series)
print("Time Series Features:", ts.extract_features())
