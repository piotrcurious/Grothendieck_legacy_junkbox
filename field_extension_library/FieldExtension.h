/**
 * FieldExtension.h
 * A compact library for ESP32 Arduino to improve numerical precision and stability
 * using field extensions over transcendental numbers.
 * 
 * This library implements field extensions based on combinations of transcendental numbers
 * like π, e, and √2 to provide higher precision arithmetic beyond standard floats.
 */

#ifndef FIELD_EXTENSION_H
#define FIELD_EXTENSION_H

#include <Arduino.h>
#include <math.h>

/**
 * Class representing an element in a field extension over transcendental numbers.
 * Each element is represented as a linear combination of basis elements:
 * a + b·π + c·e + d·√2 + e·π² + f·e·π + ...
 * 
 * The number of terms and which transcendental numbers to use can be configured
 * through template parameters.
 */
template<size_t N = 4>
class FieldElement {
    static_assert(N <= 32, "FieldElement currently supports up to 32 basis elements.");
private:
    float coefficients[N];
    
    // Basis elements information - which transcendental number combinations
    // are represented by each coefficient index. We support up to 32 basis elements.
    static const uint8_t BASIS_ELEMENTS[32][3];

    // Multiplication lookup table: [i][j] = {targetTerm, multiplier}
    struct ProductEntry { int8_t term; float multiplier; };
    static const ProductEntry PRODUCT_TABLE[32][32];
    
    // Pre-calculated values of common transcendental number powers
    static constexpr float _PI = 3.14159265358979323846f;
    static constexpr float _E = 2.71828182845904523536f;
    static constexpr float _SQRT2 = 1.41421356237309504880f;
    static constexpr float _PI2 = _PI * _PI;
    static constexpr float _E2 = _E * _E;
    static constexpr float _PI3 = _PI2 * _PI;
    static constexpr float _E3 = _E2 * _E;
    
public:
    /**
     * Default constructor - creates the element 0
     */
    FieldElement() {
        for (size_t i = 0; i < N; i++) {
            coefficients[i] = 0.0f;
        }
    }
    
    /**
     * Constructor from a regular float - creates the element with just the constant term
     */
    FieldElement(float value) {
        coefficients[0] = value;
        for (size_t i = 1; i < N; i++) {
            coefficients[i] = 0.0f;
        }
    }
    
    /**
     * Constructor with all coefficients
     */
    FieldElement(const float values[N]) {
        for (size_t i = 0; i < N; i++) {
            coefficients[i] = values[i];
        }
    }

    /**
     * Get coefficient at index
     */
    float getCoefficient(size_t index) const {
        if (index < N) {
            return coefficients[index];
        }
        return 0.0f;
    }
    
    /**
     * Set coefficient at index
     */
    void setCoefficient(size_t index, float value) {
        if (index < N) {
            coefficients[index] = value;
        }
    }

    /**
     * Addition operator
     */
    FieldElement operator+(const FieldElement& other) const {
        FieldElement result;
        for (size_t i = 0; i < N; i++) {
            result.coefficients[i] = coefficients[i] + other.coefficients[i];
        }
        return result;
    }

    /**
     * Subtraction operator
     */
    FieldElement operator-(const FieldElement& other) const {
        FieldElement result;
        for (size_t i = 0; i < N; i++) {
            result.coefficients[i] = coefficients[i] - other.coefficients[i];
        }
        return result;
    }

    /**
     * Multiplication operator - optimized using a precomputed product table
     */
    FieldElement operator*(const FieldElement& other) const {
        FieldElement result;
        
        for (size_t i = 0; i < N; i++) {
            float ci = coefficients[i];
            if (fabs(ci) < 1e-12) continue;
            for (size_t j = 0; j < N; j++) {
                float cj = other.coefficients[j];
                if (fabs(cj) < 1e-12) continue;

                float coeffProduct = ci * cj;
                ProductEntry entry = PRODUCT_TABLE[i][j];
                
                if (entry.term >= 0 && static_cast<size_t>(entry.term) < N) {
                    result.coefficients[entry.term] += coeffProduct * entry.multiplier;
                } else {
                    // Entry with negative term index requires fallback to systematic computation
                    uint8_t targetPowers[3];
                    float multiplier;
                    int resTerm = computeProductTerm(i, j, multiplier, targetPowers);
                    if (resTerm >= 0 && static_cast<size_t>(resTerm) < N) {
                        result.coefficients[resTerm] += coeffProduct * multiplier;
                    } else {
                        approximateOutOfBasisTerm(result, coeffProduct * multiplier, targetPowers);
                    }
                }
            }
        }
        
        return result;
    }
    
    /**
     * Division operator - implemented using Newton-Raphson approximation
     */
    FieldElement operator/(const FieldElement& other) const {
        // Check if divisor is close to zero
        if (other.norm() < 1e-6) {
            // Division by zero - return something indicating an error
            FieldElement result;
            result.coefficients[0] = INFINITY;
            return result;
        }
        
        // Start with a good initial guess using floating point division of evaluating values
        float initialGuess = 1.0f / other.toFloat();
        FieldElement x(initialGuess);
        
        // Newton-Raphson iterations for division: x = x + x * (1 - other * x)
        for (int iter = 0; iter < 4; iter++) {  // 4 iterations for better precision
            FieldElement temp = other * x;
            FieldElement one(1.0f);
            FieldElement error = one - temp;
            FieldElement correction = x * error;
            x = x + correction;
        }
        
        return (*this) * x;
    }
    
    /**
     * Calculate the Euclidean norm (magnitude) of the field element
     */
    float norm() const {
        float sumSquares = 0.0f;
        for (size_t i = 0; i < N; i++) {
            sumSquares += coefficients[i] * coefficients[i];
        }
        return sqrt(sumSquares);
    }
    
    /**
     * Convert back to a standard float (with loss of precision)
     * This evaluates the linear combination at the actual values
     * of the transcendental numbers.
     */
    float toFloat() const {
        float result = coefficients[0];  // Constant term
        
        // Add contributions from each basis element
        for (size_t i = 1; i < N; i++) {
            result += coefficients[i] * evaluateBasisElement(i);
        }
        
        return result;
    }
    
    /**
     * Static method to create special constants
     */
    static FieldElement pi() {
        FieldElement result;
        // Set the coefficient for π to 1.0
        result.setCoefficient(1, 1.0f);
        return result;
    }
    
    static FieldElement e() {
        FieldElement result;
        // Set the coefficient for e to 1.0
        result.setCoefficient(2, 1.0f);
        return result;
    }
    
    static FieldElement sqrt2() {
        FieldElement result;
        // Set the coefficient for √2 to 1.0
        result.setCoefficient(3, 1.0f);
        return result;
    }
    
    /**
     * Evaluate standard functions on field elements
     */
    friend FieldElement sin(const FieldElement& x) {
        // For simple elements, we can use direct formulas
        if (isSimpleElement(x)) {
            return sinSimpleElement(x);
        }
        
        // For complex elements, use Taylor series approximation
        return sinTaylor(x);
    }
    
    friend FieldElement cos(const FieldElement& x) {
        // For simple elements, we can use direct formulas
        if (isSimpleElement(x)) {
            return cosSimpleElement(x);
        }
        
        // For complex elements, use Taylor series approximation
        return cosTaylor(x);
    }
    
    friend FieldElement exp(const FieldElement& x) {
        // For simple elements, we can use direct formulas
        if (isSimpleElement(x)) {
            return expSimpleElement(x);
        }
        
        // For complex elements, use Taylor series approximation
        return expTaylor(x);
    }
    
    friend FieldElement log(const FieldElement& x) {
        // For simple elements, we can use direct formulas
        if (isSimpleElement(x)) {
            return logSimpleElement(x);
        }
        
        // For complex elements, use Taylor series approximation
        return logTaylor(x);
    }

    friend FieldElement sqrt(const FieldElement& x) {
        float val = x.toFloat();
        if (val < 0) return FieldElement(NAN);
        if (val == 0) return FieldElement(0.0f);

        // Newton's method: y = 0.5 * (y + x/y)
        FieldElement y(std::sqrt(val));
        for (int i = 0; i < 4; i++) {
            y = (y + x / y) * 0.5f;
        }
        return y;
    }

    friend FieldElement atan(const FieldElement& x) {
        float val = x.toFloat();
        if (std::isnan(val)) return FieldElement(NAN);

        // Use threshold 0.6 to avoid slow convergence and recursion issues at 1.0
        if (std::abs(val) < 0.6f) {
            FieldElement result = x;
            FieldElement x2 = x * x;
            FieldElement term = x;
            for (int i = 1; i <= 8; i++) {
                term = term * x2;
                float sign = (i % 2 == 0) ? 1.0f : -1.0f;
                result = result + term * (sign / (2.0f * i + 1.0f));
            }
            return result;
        } else {
            // Range reduction using atan(x) = atan(c) + atan((x-c)/(1+xc))
            // To keep it simple, we can use atan(x) = pi/2 - atan(1/x) for |x| > 1
            // but for x near 1, we still need something.
            if (val > 1.0f) {
                return FieldElement(_PI * 0.5f) - atan(FieldElement(1.0f) / x);
            } else if (val < -1.0f) {
                return FieldElement(-_PI * 0.5f) - atan(FieldElement(1.0f) / x);
            } else {
                // Near 1 or -1: use atan(x) = 2 * atan(x / (1 + sqrt(1 + x^2)))
                // This will bring the argument close to 0.414 for x=1
                return atan(x / (FieldElement(1.0f) + sqrt(FieldElement(1.0f) + x * x))) * 2.0f;
            }
        }
    }

    friend FieldElement asin(const FieldElement& x) {
        float val = x.toFloat();
        if (std::abs(val) > 1.0f) return FieldElement(NAN);
        if (std::abs(val) == 1.0f) return FieldElement(val * _PI * 0.5f);
        // asin(x) = atan(x / sqrt(1 - x^2))
        return atan(x / sqrt(FieldElement(1.0f) - x * x));
    }

    friend FieldElement acos(const FieldElement& x) {
        // acos(x) = pi/2 - asin(x)
        return FieldElement(_PI * 0.5f) - asin(x);
    }

    friend FieldElement tan(const FieldElement& x) {
        return sin(x) / cos(x);
    }

    friend FieldElement sinh(const FieldElement& x) {
        FieldElement ex = exp(x);
        return (ex - FieldElement(1.0f) / ex) * 0.5f;
    }

    friend FieldElement cosh(const FieldElement& x) {
        FieldElement ex = exp(x);
        return (ex + FieldElement(1.0f) / ex) * 0.5f;
    }

    friend FieldElement tanh(const FieldElement& x) {
        FieldElement ex = exp(x);
        FieldElement enx = FieldElement(1.0f) / ex;
        return (ex - enx) / (ex + enx);
    }

    friend FieldElement asinh(const FieldElement& x) {
        // asinh(x) = log(x + sqrt(x^2 + 1))
        return log(x + sqrt(x * x + FieldElement(1.0f)));
    }

    friend FieldElement acosh(const FieldElement& x) {
        // acosh(x) = log(x + sqrt(x^2 - 1))
        if (x.toFloat() < 1.0f) return FieldElement(NAN);
        return log(x + sqrt(x * x - FieldElement(1.0f)));
    }

    friend FieldElement atanh(const FieldElement& x) {
        // atanh(x) = 0.5 * log((1+x)/(1-x))
        float v = x.toFloat();
        if (std::abs(v) >= 1.0f) return FieldElement(NAN);
        return log((FieldElement(1.0f) + x) / (FieldElement(1.0f) - x)) * 0.5f;
    }

    friend FieldElement pow(const FieldElement& x, float y) {
        float val = x.toFloat();
        if (val < 0 && std::floor(y) != y) return FieldElement(NAN);
        if (val == 0) return (y == 0) ? FieldElement(1.0f) : FieldElement(0.0f);

        // Optimization for integer powers
        if (std::floor(y) == y && std::abs(y) <= 10.0f) {
            int iy = (int)y;
            if (iy == 0) return FieldElement(1.0f);
            FieldElement base = (iy > 0) ? x : FieldElement(1.0f) / x;
            int ay = std::abs(iy);
            FieldElement res(1.0f);
            for (int i = 0; i < ay; i++) res = res * base;
            return res;
        }

        // Use exp(y * log(x)) for general power
        return exp(log(x) * y);
    }

    friend FieldElement pow(const FieldElement& x, const FieldElement& y) {
        float valX = x.toFloat();
        if (valX < 0) return FieldElement(NAN);
        if (valX == 0) return (y.toFloat() == 0) ? FieldElement(1.0f) : FieldElement(0.0f);
        return exp(log(x) * y);
    }

    friend FieldElement atan2(const FieldElement& y, const FieldElement& x) {
        float vx = x.toFloat();
        float vy = y.toFloat();
        if (vx > 0) return atan(y / x);
        if (vx < 0 && vy >= 0) return atan(y / x) + FieldElement(_PI);
        if (vx < 0 && vy < 0) return atan(y / x) - FieldElement(_PI);
        if (vx == 0 && vy > 0) return FieldElement(_PI * 0.5f);
        if (vx == 0 && vy < 0) return FieldElement(-_PI * 0.5f);
        return FieldElement(0.0f); // atan2(0,0) is undefined, return 0
    }
    
private:
    /**
     * Determine which basis term results from multiplying terms i and j
     */
    int computeProductTerm(size_t i, size_t j, float& multiplier, uint8_t targetPowers[3]) const {
        multiplier = 1.0f;
        if (i >= 32 || j >= 32) return -1;

        for (int k = 0; k < 3; k++) {
            targetPowers[k] = BASIS_ELEMENTS[i][k] + BASIS_ELEMENTS[j][k];
        }
        
        // Handle relations: sqrt(2)^2 = 2
        while (targetPowers[2] >= 2) {
            targetPowers[2] -= 2;
            multiplier *= 2.0f;
        }

        for (size_t k = 0; k < N; k++) {
            if (BASIS_ELEMENTS[k][0] == targetPowers[0] &&
                BASIS_ELEMENTS[k][1] == targetPowers[1] &&
                BASIS_ELEMENTS[k][2] == targetPowers[2]) {
                return (int)k;
            }
        }
        return -1;
    }
    
    /**
     * Approximate an out-of-basis term by evaluating it to float and adding to constant term
     */
    void approximateOutOfBasisTerm(FieldElement& result, float coefficient, 
                                  const uint8_t powers[3]) const {
        float val = coefficient;
        if (powers[0] > 0) val *= pow(_PI, (float)powers[0]);
        if (powers[1] > 0) val *= pow(_E, (float)powers[1]);
        if (powers[2] > 0) val *= pow(_SQRT2, (float)powers[2]);
        result.coefficients[0] += val;
    }
    
    /**
     * Evaluate a basis element at the actual transcendental values
     */
    float evaluateBasisElement(size_t index) const {
        if (index >= 32) return 0.0f;
        float val = 1.0f;

        switch (BASIS_ELEMENTS[index][0]) {
            case 1: val *= _PI; break;
            case 2: val *= _PI2; break;
            case 3: val *= _PI3; break;
        }
        switch (BASIS_ELEMENTS[index][1]) {
            case 1: val *= _E; break;
            case 2: val *= _E2; break;
            case 3: val *= _E3; break;
        }
        if (BASIS_ELEMENTS[index][2] == 1) val *= _SQRT2;

        return val;
    }
    
    /**
     * Check if an element is simple (mainly coefficients for basic elements)
     */
    static bool isSimpleElement(const FieldElement& x) {
        int nonZeroCount = 0;
        for (size_t i = 0; i < N; i++) {
            if (fabs(x.coefficients[i]) > 1e-6) {
                nonZeroCount++;
            }
        }
        // Consider simple if at most 2 coefficients are non-zero
        return nonZeroCount <= 2;
    }
    
    /**
     * Implement sin for simple elements
     */
    static FieldElement sinSimpleElement(const FieldElement& x) {
        // Symbolic shortcuts
        if (x.coefficients[1] == 1.0f && x.norm() == 1.0f) return FieldElement(0.0f); // sin(pi)
        if (x.coefficients[1] == 0.5f && x.norm() == 0.5f) return FieldElement(1.0f); // sin(pi/2)
        if (x.coefficients[1] == -0.5f && x.norm() == 0.5f) return FieldElement(-1.0f); // sin(-pi/2)

        FieldElement result;
        result.coefficients[0] = sin(x.toFloat());
        return result;
    }
    
    /**
     * Implement cos for simple elements
     */
    static FieldElement cosSimpleElement(const FieldElement& x) {
        // Symbolic shortcuts
        if (x.coefficients[1] == 1.0f && x.norm() == 1.0f) return FieldElement(-1.0f); // cos(pi)
        if (x.coefficients[1] == 0.5f && x.norm() == 0.5f) return FieldElement(0.0f); // cos(pi/2)
        if (x.coefficients[1] == -0.5f && x.norm() == 0.5f) return FieldElement(0.0f); // cos(-pi/2)

        FieldElement result;
        result.coefficients[0] = cos(x.toFloat());
        return result;
    }
    
    /**
     * Implement exp for simple elements
     */
    static FieldElement expSimpleElement(const FieldElement& x) {
        // Symbolic shortcuts
        if (x.coefficients[0] == 1.0f && x.norm() == 1.0f) return FieldElement::e(); // exp(1) = e

        FieldElement result;
        result.coefficients[0] = exp(x.toFloat());
        return result;
    }
    
    /**
     * Implement log for simple elements
     */
    static FieldElement logSimpleElement(const FieldElement& x) {
        // Symbolic shortcuts
        if (x.coefficients[2] == 1.0f && x.norm() == 1.0f) return FieldElement(1.0f); // log(e) = 1

        FieldElement result;
        result.coefficients[0] = log(x.toFloat());
        return result;
    }
    
    /**
     * Taylor series approximation for sin
     */
    static FieldElement sinTaylor(const FieldElement& x) {
        // Range reduction to [-pi, pi]
        float val = x.toFloat();
        int periods = (int)(val / (2.0f * _PI));
        FieldElement reducedX = x - FieldElement(_PI * 2.0f * periods);

        // If still outside [-pi, pi], adjust
        float rVal = reducedX.toFloat();
        if (rVal > _PI) reducedX = reducedX - FieldElement(_PI * 2.0f);
        else if (rVal < -_PI) reducedX = reducedX + FieldElement(_PI * 2.0f);

        // Check for symbolic shortcuts
        rVal = reducedX.toFloat();
        if (fabs(rVal) < 1e-6) return FieldElement(0.0f);
        if (fabs(rVal - _PI) < 1e-6 || fabs(rVal + _PI) < 1e-6) return FieldElement(0.0f);
        if (fabs(rVal - _PI*0.5f) < 1e-6) return FieldElement(1.0f);
        if (fabs(rVal + _PI*0.5f) < 1e-6) return FieldElement(-1.0f);

        FieldElement result;
        FieldElement xSquared = reducedX * reducedX;
        FieldElement term = reducedX;  // First term: x
        
        // Add terms: x - x^3/3! + x^5/5! - x^7/7! + x^9/9!
        result = result + term;
        
        term = term * xSquared * (-1.0f/6.0f);
        result = result + term;
        
        term = term * xSquared * (1.0f/20.0f);
        result = result + term;
        
        term = term * xSquared * (-1.0f/42.0f);
        result = result + term;

        term = term * xSquared * (1.0f/72.0f);
        result = result + term;
        
        return result;
    }
    
    /**
     * Taylor series approximation for cos
     */
    static FieldElement cosTaylor(const FieldElement& x) {
        // Range reduction to [-pi, pi]
        float val = x.toFloat();
        int periods = (int)(val / (2.0f * _PI));
        FieldElement reducedX = x - FieldElement(_PI * 2.0f * periods);

        float rVal = reducedX.toFloat();
        if (rVal > _PI) reducedX = reducedX - FieldElement(_PI * 2.0f);
        else if (rVal < -_PI) reducedX = reducedX + FieldElement(_PI * 2.0f);

        // Check for symbolic shortcuts
        rVal = reducedX.toFloat();
        if (fabs(rVal) < 1e-6) return FieldElement(1.0f);
        if (fabs(rVal - _PI) < 1e-6 || fabs(rVal + _PI) < 1e-6) return FieldElement(-1.0f);
        if (fabs(rVal - _PI*0.5f) < 1e-6 || fabs(rVal + _PI*0.5f) < 1e-6) return FieldElement(0.0f);

        FieldElement result(1.0f);  // First term: 1
        FieldElement xSquared = reducedX * reducedX;
        FieldElement term(1.0f);
        
        // 1 - x^2/2! + x^4/4! - x^6/6! + x^8/8!
        term = term * xSquared * (-1.0f/2.0f);
        result = result + term;
        
        term = term * xSquared * (1.0f/12.0f);
        result = result + term;
        
        term = term * xSquared * (-1.0f/30.0f);
        result = result + term;

        term = term * xSquared * (1.0f/56.0f);
        result = result + term;
        
        return result;
    }
    
    /**
     * Taylor series approximation for exp
     */
    static FieldElement expTaylor(const FieldElement& x) {
        // Range reduction: exp(x) = exp(x - k) * e^k
        float val = x.toFloat();
        int k = (int)round(val);
        FieldElement reducedX = x - FieldElement((float)k);

        FieldElement result(1.0f);  // First term: 1
        FieldElement term(1.0f);
        FieldElement xPower = reducedX;
        
        // Add terms: 1 + x + x^2/2! + x^3/3! + x^4/4! + x^5/5! + x^6/6!
        for (int i = 1; i <= 6; i++) {
            term = xPower * (1.0f / factorial(i));
            result = result + term;
            xPower = xPower * reducedX;
        }

        // Multiply by e^k
        if (k != 0) {
            // We need a way to multiply by e^k symbolically if possible
            // For now, if N >= 3 we have e as term 2.
            // But e^k might not be in basis.
            // Let's use a simple loop of multiplications if k is small and positive
            if (k > 0 && k <= 3 && N >= 3) {
                FieldElement eElem = FieldElement::e();
                for (int i = 0; i < k; i++) result = result * eElem;
            } else {
                result = result * exp((float)k);
            }
        }
        
        return result;
    }
    
    /**
     * Taylor series approximation for log(x)
     */
    static FieldElement logTaylor(const FieldElement& x) {
        // Range reduction: log(x) = log(x/e^k) + k
        float val = x.toFloat();
        if (val <= 0) return FieldElement(NAN);

        int k = (int)round(log(val));
        // x_reduced = x * e^-k
        FieldElement xReduced = x * exp((float)-k);

        // Use log(1+y) where y = x_reduced - 1
        FieldElement one(1.0f);
        FieldElement y = xReduced - one;
        
        // Taylor series for log(1+y): y - y^2/2 + y^3/3 - y^4/4 + y^5/5
        FieldElement result = y;
        FieldElement yPower = y * y;
        
        result = result - yPower * 0.5f;
        
        yPower = yPower * y;
        result = result + yPower * (1.0f/3.0f);
        
        yPower = yPower * y;
        result = result - yPower * 0.25f;

        yPower = yPower * y;
        result = result + yPower * 0.2f;

        if (k != 0) {
            result = result + FieldElement((float)k);
        }
        
        return result;
    }
    
    /**
     * Calculate factorial
     */
    static float factorial(int n) {
        float result = 1.0f;
        for (int i = 2; i <= n; i++) {
            result *= i;
        }
        return result;
    }
};

/**
 * Definition of basis elements
 * Each row represents a basis element, and the values indicate powers of
 * transcendental numbers: [power of π, power of e, power of √2]
 */
template<size_t N>
const uint8_t FieldElement<N>::BASIS_ELEMENTS[32][3] = {
    {0, 0, 0},    {1, 0, 0},    {0, 1, 0},    {0, 0, 1},
    {2, 0, 0},    {0, 2, 0},    {1, 1, 0},    {1, 0, 1},
    {0, 1, 1},    {3, 0, 0},    {0, 3, 0},    {2, 1, 0},
    {1, 2, 0},    {2, 0, 1},    {0, 2, 1},    {1, 1, 1},
    {0, 3, 1},    {1, 2, 1},    {1, 3, 0},    {1, 3, 1},
    {2, 1, 1},    {2, 2, 0},    {2, 2, 1},    {2, 3, 0},
    {2, 3, 1},    {3, 0, 1},    {3, 1, 0},    {3, 1, 1},
    {3, 2, 0},    {3, 2, 1},    {3, 3, 0},    {3, 3, 1},
};

template<size_t N>
const typename FieldElement<N>::ProductEntry FieldElement<N>::PRODUCT_TABLE[32][32] = {
    /*  0 */ {{0,1}, {1,1}, {2,1}, {3,1}, {4,1}, {5,1}, {6,1}, {7,1}, {8,1}, {9,1}, {10,1}, {11,1}, {12,1}, {13,1}, {14,1}, {15,1}, {16,1}, {17,1}, {18,1}, {19,1}, {20,1}, {21,1}, {22,1}, {23,1}, {24,1}, {25,1}, {26,1}, {27,1}, {28,1}, {29,1}, {30,1}, {31,1}},
    /*  1 */ {{1,1}, {4,1}, {6,1}, {7,1}, {9,1}, {12,1}, {11,1}, {13,1}, {15,1}, {-1,1}, {18,1}, {26,1}, {21,1}, {25,1}, {17,1}, {20,1}, {19,1}, {22,1}, {23,1}, {24,1}, {27,1}, {28,1}, {29,1}, {30,1}, {31,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}},
    /*  2 */ {{2,1}, {6,1}, {5,1}, {8,1}, {11,1}, {10,1}, {12,1}, {15,1}, {14,1}, {26,1}, {-1,1}, {21,1}, {18,1}, {20,1}, {16,1}, {17,1}, {-1,1}, {19,1}, {-1,1}, {-1,1}, {22,1}, {23,1}, {24,1}, {-1,1}, {-1,1}, {27,1}, {28,1}, {29,1}, {30,1}, {31,1}, {-1,1}, {-1,1}},
    /*  3 */ {{3,1}, {7,1}, {8,1}, {0,2}, {13,1}, {14,1}, {15,1}, {1,2}, {2,2}, {25,1}, {16,1}, {20,1}, {17,1}, {4,2}, {5,2}, {6,2}, {10,2}, {12,2}, {19,1}, {18,2}, {11,2}, {22,1}, {21,2}, {24,1}, {23,2}, {9,2}, {27,1}, {26,2}, {29,1}, {28,2}, {31,1}, {30,2}},
    /*  4 */ {{4,1}, {9,1}, {11,1}, {13,1}, {-1,1}, {21,1}, {26,1}, {25,1}, {20,1}, {-1,1}, {23,1}, {-1,1}, {28,1}, {-1,1}, {22,1}, {27,1}, {24,1}, {29,1}, {30,1}, {31,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}},
    /*  5 */ {{5,1}, {12,1}, {10,1}, {14,1}, {21,1}, {-1,1}, {18,1}, {17,1}, {16,1}, {28,1}, {-1,1}, {23,1}, {-1,1}, {22,1}, {-1,1}, {19,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {24,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {29,1}, {30,1}, {31,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}},
    /*  6 */ {{6,1}, {11,1}, {12,1}, {15,1}, {26,1}, {18,1}, {21,1}, {20,1}, {17,1}, {-1,1}, {-1,1}, {28,1}, {23,1}, {27,1}, {19,1}, {22,1}, {-1,1}, {24,1}, {-1,1}, {-1,1}, {29,1}, {30,1}, {31,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}},
    /*  7 */ {{7,1}, {13,1}, {15,1}, {1,2}, {25,1}, {17,1}, {20,1}, {4,2}, {6,2}, {-1,1}, {19,1}, {27,1}, {22,1}, {9,2}, {12,2}, {11,2}, {18,2}, {21,2}, {24,1}, {23,2}, {26,2}, {29,1}, {28,2}, {31,1}, {30,2}, {-1,2}, {-1,1}, {-1,2}, {-1,1}, {-1,2}, {-1,1}, {-1,2}},
    /*  8 */ {{8,1}, {15,1}, {14,1}, {2,2}, {20,1}, {16,1}, {17,1}, {6,2}, {5,2}, {27,1}, {-1,1}, {22,1}, {19,1}, {11,2}, {10,2}, {12,2}, {-1,2}, {18,2}, {-1,1}, {-1,2}, {21,2}, {24,1}, {23,2}, {-1,1}, {-1,2}, {26,2}, {29,1}, {28,2}, {31,1}, {30,2}, {-1,1}, {-1,2}},
    /*  9 */ {{9,1}, {-1,1}, {26,1}, {25,1}, {-1,1}, {28,1}, {-1,1}, {-1,1}, {27,1}, {-1,1}, {30,1}, {-1,1}, {-1,1}, {-1,1}, {29,1}, {-1,1}, {31,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}},
    /* 10 */ {{10,1}, {18,1}, {-1,1}, {16,1}, {23,1}, {-1,1}, {-1,1}, {19,1}, {-1,1}, {30,1}, {-1,1}, {-1,1}, {-1,1}, {24,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {31,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}},
    /* 11 */ {{11,1}, {26,1}, {21,1}, {20,1}, {-1,1}, {23,1}, {28,1}, {27,1}, {22,1}, {-1,1}, {-1,1}, {-1,1}, {30,1}, {-1,1}, {24,1}, {29,1}, {-1,1}, {31,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}},
    /* 12 */ {{12,1}, {21,1}, {18,1}, {17,1}, {28,1}, {-1,1}, {23,1}, {22,1}, {19,1}, {-1,1}, {-1,1}, {30,1}, {-1,1}, {29,1}, {-1,1}, {24,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {31,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}},
    /* 13 */ {{13,1}, {25,1}, {20,1}, {4,2}, {-1,1}, {22,1}, {27,1}, {9,2}, {11,2}, {-1,1}, {24,1}, {-1,1}, {29,1}, {-1,2}, {21,2}, {26,2}, {23,2}, {28,2}, {31,1}, {30,2}, {-1,2}, {-1,1}, {-1,2}, {-1,1}, {-1,2}, {-1,2}, {-1,1}, {-1,2}, {-1,1}, {-1,2}, {-1,1}, {-1,2}},
    /* 14 */ {{14,1}, {17,1}, {16,1}, {5,2}, {22,1}, {-1,1}, {19,1}, {12,2}, {10,2}, {29,1}, {-1,1}, {24,1}, {-1,1}, {21,2}, {-1,2}, {18,2}, {-1,2}, {-1,2}, {-1,1}, {-1,2}, {23,2}, {-1,1}, {-1,2}, {-1,1}, {-1,2}, {28,2}, {31,1}, {30,2}, {-1,1}, {-1,2}, {-1,1}, {-1,2}},
    /* 15 */ {{15,1}, {20,1}, {17,1}, {6,2}, {27,1}, {19,1}, {22,1}, {11,2}, {12,2}, {-1,1}, {-1,1}, {29,1}, {24,1}, {26,2}, {18,2}, {21,2}, {-1,2}, {23,2}, {-1,1}, {-1,2}, {28,2}, {31,1}, {30,2}, {-1,1}, {-1,2}, {-1,2}, {-1,1}, {-1,2}, {-1,1}, {-1,2}, {-1,1}, {-1,2}},
    /* 16 */ {{16,1}, {19,1}, {-1,1}, {10,2}, {24,1}, {-1,1}, {-1,1}, {18,2}, {-1,2}, {31,1}, {-1,1}, {-1,1}, {-1,1}, {23,2}, {-1,2}, {-1,2}, {-1,2}, {-1,2}, {-1,1}, {-1,2}, {30,2}, {-1,1}, {-1,2}, {-1,1}, {-1,2}, {28,2}, {31,1}, {30,2}, {-1,1}, {-1,2}, {-1,1}, {-1,2}},
    /* 17 */ {{17,1}, {22,1}, {19,1}, {12,2}, {29,1}, {-1,1}, {24,1}, {21,2}, {18,2}, {-1,1}, {-1,1}, {31,1}, {-1,1}, {28,2}, {-1,2}, {23,2}, {-1,2}, {-1,2}, {-1,1}, {-1,2}, {30,2}, {-1,1}, {-1,2}, {-1,1}, {-1,2}, {-1,2}, {-1,1}, {-1,2}, {-1,1}, {-1,2}, {-1,1}, {-1,2}},
    /* 18 */ {{18,1}, {23,1}, {-1,1}, {19,1}, {30,1}, {-1,1}, {-1,1}, {24,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {31,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}},
    /* 19 */ {{19,1}, {24,1}, {-1,1}, {18,2}, {31,1}, {-1,1}, {-1,1}, {23,2}, {-1,2}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {30,2}, {-1,2}, {-1,2}, {-1,2}, {-1,2}, {-1,1}, {-1,2}, {30,2}, {-1,1}, {-1,2}, {-1,1}, {-1,2}, {-1,2}, {-1,1}, {-1,2}, {-1,1}, {-1,2}, {-1,1}, {-1,2}},
    /* 20 */ {{20,1}, {27,1}, {22,1}, {11,2}, {-1,1}, {24,1}, {29,1}, {26,2}, {21,2}, {-1,1}, {-1,1}, {-1,1}, {31,1}, {-1,2}, {23,2}, {28,2}, {-1,2}, {30,2}, {-1,1}, {-1,2}, {-1,2}, {-1,1}, {-1,2}, {-1,1}, {-1,2}, {-1,2}, {-1,1}, {-1,2}, {-1,1}, {-1,2}, {-1,1}, {-1,2}},
    /* 21 */ {{21,1}, {28,1}, {23,1}, {22,1}, {-1,1}, {-1,1}, {30,1}, {29,1}, {24,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {31,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}},
    /* 22 */ {{22,1}, {29,1}, {24,1}, {21,2}, {-1,1}, {-1,1}, {31,1}, {28,2}, {23,2}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,2}, {-1,2}, {30,2}, {-1,2}, {-1,2}, {-1,1}, {-1,2}, {-1,2}, {-1,1}, {-1,2}, {-1,1}, {-1,2}, {-1,2}, {-1,1}, {-1,2}, {-1,1}, {-1,2}, {-1,1}, {-1,2}},
    /* 23 */ {{23,1}, {30,1}, {-1,1}, {24,1}, {-1,1}, {-1,1}, {-1,1}, {31,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}},
    /* 24 */ {{24,1}, {31,1}, {-1,1}, {23,2}, {-1,1}, {-1,1}, {-1,1}, {30,2}, {-1,2}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,2}, {-1,2}, {-1,2}, {-1,2}, {-1,2}, {-1,1}, {-1,2}, {-1,2}, {-1,1}, {-1,2}, {-1,1}, {-1,2}, {-1,2}, {-1,1}, {-1,2}, {-1,1}, {-1,2}, {-1,1}, {-1,2}},
    /* 25 */ {{25,1}, {-1,1}, {27,1}, {9,2}, {-1,1}, {29,1}, {-1,1}, {-1,2}, {26,2}, {-1,1}, {31,1}, {-1,1}, {-1,1}, {-1,2}, {28,2}, {-1,2}, {30,2}, {-1,2}, {-1,1}, {-1,2}, {-1,2}, {-1,1}, {-1,2}, {-1,1}, {-1,2}, {-1,2}, {-1,1}, {-1,2}, {-1,1}, {-1,2}, {-1,1}, {-1,2}},
    /* 26 */ {{26,1}, {-1,1}, {28,1}, {27,1}, {-1,1}, {30,1}, {-1,1}, {-1,1}, {29,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {31,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}},
    /* 27 */ {{27,1}, {-1,1}, {29,1}, {26,2}, {-1,1}, {31,1}, {-1,1}, {-1,2}, {28,2}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,2}, {30,2}, {-1,2}, {-1,2}, {-1,2}, {-1,1}, {-1,2}, {-1,2}, {-1,1}, {-1,2}, {-1,1}, {-1,2}, {-1,2}, {-1,1}, {-1,2}, {-1,1}, {-1,2}, {-1,1}, {-1,2}},
    /* 28 */ {{28,1}, {-1,1}, {30,1}, {29,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {31,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}},
    /* 29 */ {{29,1}, {-1,1}, {31,1}, {28,2}, {-1,1}, {-1,1}, {-1,1}, {-1,2}, {30,2}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,2}, {-1,2}, {-1,2}, {-1,2}, {-1,2}, {-1,1}, {-1,2}, {-1,2}, {-1,1}, {-1,2}, {-1,1}, {-1,2}, {-1,2}, {-1,1}, {-1,2}, {-1,1}, {-1,2}, {-1,1}, {-1,2}},
    /* 30 */ {{30,1}, {-1,1}, {-1,1}, {31,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,1}},
    /* 31 */ {{31,1}, {-1,1}, {-1,1}, {30,2}, {-1,1}, {-1,1}, {-1,1}, {-1,2}, {-1,2}, {-1,1}, {-1,1}, {-1,1}, {-1,1}, {-1,2}, {-1,2}, {-1,2}, {-1,2}, {-1,2}, {-1,1}, {-1,2}, {-1,2}, {-1,1}, {-1,2}, {-1,1}, {-1,2}, {-1,2}, {-1,1}, {-1,2}, {-1,1}, {-1,2}, {-1,1}, {-1,2}},
};

/**
 * Type aliases for common field extension sizes
 */
using FieldElement4 = FieldElement<4>;   // Basic field with [1, π, e, √2]
using FieldElement8 = FieldElement<8>;   // Extended field with more combinations
using FieldElement16 = FieldElement<16>; // Advanced field with many combinations

#endif // FIELD_EXTENSION_H
