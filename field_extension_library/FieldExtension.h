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
private:
    float coefficients[N];
    
    // Basis elements information - which transcendental number combinations
    // are represented by each coefficient index. We support up to 16 basis elements.
    static const uint8_t BASIS_ELEMENTS[16][3];
    
    // Pre-calculated values of common transcendental number powers
    static constexpr float _PI = 3.14159265358979323846;
    static constexpr float _E = 2.71828182845904523536;
    static constexpr float _SQRT2 = 1.41421356237309504880;
    static constexpr float _PI2 = _PI * _PI;
    static constexpr float _E2 = _E * _E;
    
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
     * Multiplication operator - this is the complex part because we need
     * to handle multiplication of the transcendental basis elements
     */
    FieldElement operator*(const FieldElement& other) const {
        FieldElement result;
        
        // For each pair of terms in the two field elements
        for (size_t i = 0; i < N; i++) {
            if (fabs(coefficients[i]) < 1e-9) continue;
            for (size_t j = 0; j < N; j++) {
                if (fabs(other.coefficients[j]) < 1e-9) continue;

                // Multiply the coefficients
                float coeffProduct = coefficients[i] * other.coefficients[j];
                
                // Compute which result term this product contributes to
                float multiplier = 1.0f;
                uint8_t targetPowers[3];
                int resultTerm = computeProductTerm(i, j, multiplier, targetPowers);
                
                // If the result is part of our basis, add it to the appropriate coefficient
                if (resultTerm >= 0 && static_cast<size_t>(resultTerm) < N) {
                    result.coefficients[resultTerm] += coeffProduct * multiplier;
                }
                // If the resultTerm is outside our basis, we'll approximate
                else {
                    approximateOutOfBasisTerm(result, coeffProduct * multiplier, targetPowers);
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
    
private:
    /**
     * Determine which basis term results from multiplying terms i and j
     */
    int computeProductTerm(size_t i, size_t j, float& multiplier, uint8_t targetPowers[3]) const {
        multiplier = 1.0f;
        if (i >= 16 || j >= 16) return -1;

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
        if (index >= 16) return 0.0f;
        float val = 1.0f;
        if (BASIS_ELEMENTS[index][0] > 0) val *= pow(_PI, (float)BASIS_ELEMENTS[index][0]);
        if (BASIS_ELEMENTS[index][1] > 0) val *= pow(_E, (float)BASIS_ELEMENTS[index][1]);
        if (BASIS_ELEMENTS[index][2] > 0) val *= pow(_SQRT2, (float)BASIS_ELEMENTS[index][2]);
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
        // Simple implementation for demonstration
        FieldElement result;
        result.coefficients[0] = sin(x.toFloat());
        return result;
    }
    
    /**
     * Implement cos for simple elements
     */
    static FieldElement cosSimpleElement(const FieldElement& x) {
        // Simple implementation for demonstration
        FieldElement result;
        result.coefficients[0] = cos(x.toFloat());
        return result;
    }
    
    /**
     * Implement exp for simple elements
     */
    static FieldElement expSimpleElement(const FieldElement& x) {
        // Simple implementation for demonstration
        FieldElement result;
        result.coefficients[0] = exp(x.toFloat());
        return result;
    }
    
    /**
     * Implement log for simple elements
     */
    static FieldElement logSimpleElement(const FieldElement& x) {
        // Simple implementation for demonstration
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
const uint8_t FieldElement<N>::BASIS_ELEMENTS[16][3] = {
    {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, // 0-3: 1, pi, e, sqrt2
    {2, 0, 0}, {0, 2, 0}, {1, 1, 0}, {1, 0, 1}, // 4-7: pi^2, e^2, pi*e, pi*sqrt2
    {0, 1, 1}, {3, 0, 0}, {0, 3, 0}, {2, 1, 0}, // 8-11: e*sqrt2, pi^3, e^3, pi^2*e
    {1, 2, 0}, {2, 0, 1}, {0, 2, 1}, {1, 1, 1}  // 12-15: pi*e^2, pi^2*sqrt2, e^2*sqrt2, pi*e*sqrt2
};

/**
 * Type aliases for common field extension sizes
 */
using FieldElement4 = FieldElement<4>;   // Basic field with [1, π, e, √2]
using FieldElement8 = FieldElement<8>;   // Extended field with more combinations
using FieldElement16 = FieldElement<16>; // Advanced field with many combinations

#endif // FIELD_EXTENSION_H
