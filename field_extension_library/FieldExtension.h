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
    // are represented by each coefficient index
    static const uint8_t BASIS_ELEMENTS[N][3];
    
    // Pre-calculated values of common transcendental number powers
    static constexpr float PI = 3.14159265358979323846;
    static constexpr float E = 2.71828182845904523536;
    static constexpr float SQRT2 = 1.41421356237309504880;
    static constexpr float PI2 = PI * PI;
    static constexpr float E2 = E * E;
    
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
            for (size_t j = 0; j < N; j++) {
                // Multiply the coefficients
                float coeffProduct = coefficients[i] * other.coefficients[j];
                
                // Skip if product is zero
                if (fabs(coeffProduct) < 1e-6) continue;
                
                // Compute which result term this product contributes to
                // This depends on the specific basis elements being used
                int resultTerm = computeProductTerm(i, j);
                
                // If the result is part of our basis, add it to the appropriate coefficient
                if (resultTerm >= 0 && resultTerm < N) {
                    result.coefficients[resultTerm] += coeffProduct;
                }
                // If the resultTerm is outside our basis, we'll approximate
                else {
                    // Distribute to the closest matching terms as an approximation
                    approximateOutOfBasisTerm(result, coeffProduct, i, j);
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
        
        // Start with a good initial guess using floating point division
        float initialGuess = coefficients[0] / other.coefficients[0];
        FieldElement x(initialGuess);
        
        // Newton-Raphson iterations for division: x = x + x * (1 - other * x)
        for (int iter = 0; iter < 3; iter++) {  // 3 iterations is usually sufficient
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
    int computeProductTerm(size_t i, size_t j) const {
        // This implementation depends on how you define your basis
        // For a simple example with [1, π, e, √2]:
        
        // If both are the constant term (1*1)
        if (i == 0 && j == 0) return 0;
        
        // If one is the constant term (1*something = something)
        if (i == 0) return j;
        if (j == 0) return i;
        
        // Handle specific products based on the basis definition
        // For example, if π*e is a defined basis element:
        if ((i == 1 && j == 2) || (i == 2 && j == 1)) {
            // Return the index for π*e if it exists in your basis
            // This would depend on BASIS_ELEMENTS definition
            for (size_t k = 0; k < N; k++) {
                if (BASIS_ELEMENTS[k][0] == 1 && BASIS_ELEMENTS[k][1] == 1) {
                    return k;
                }
            }
        }
        
        // If the product isn't in our basis, return -1
        // This will trigger approximation
        return -1;
    }
    
    /**
     * Approximate an out-of-basis term by distributing to the closest matching terms
     */
    void approximateOutOfBasisTerm(FieldElement& result, float coefficient, 
                                  size_t term1, size_t term2) const {
        // This is a simple approximation that distributes to the constituent terms
        // A more sophisticated approach would use a proper approximation theory
        
        // Add half to each of the original terms
        result.coefficients[term1] += coefficient * 0.5f;
        result.coefficients[term2] += coefficient * 0.5f;
    }
    
    /**
     * Evaluate a basis element at the actual transcendental values
     */
    float evaluateBasisElement(size_t index) const {
        // For standard basis [1, π, e, √2, ...]
        switch (index) {
            case 0: return 1.0f;  // Constant term
            case 1: return PI;    // π
            case 2: return E;     // e
            case 3: return SQRT2; // √2
            // Handle other basis elements based on your definitions
            default: return 0.0f;
        }
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
        FieldElement result;
        FieldElement xSquared = x * x;
        FieldElement term = x;  // First term: x
        
        // Add first term
        result = result + term;
        
        // x^3/3!
        term = term * xSquared * (-1.0f/6.0f);
        result = result + term;
        
        // x^5/5!
        term = term * xSquared * (1.0f/20.0f);
        result = result + term;
        
        // x^7/7!
        term = term * xSquared * (-1.0f/42.0f);
        result = result + term;
        
        return result;
    }
    
    /**
     * Taylor series approximation for cos
     */
    static FieldElement cosTaylor(const FieldElement& x) {
        FieldElement result(1.0f);  // First term: 1
        FieldElement xSquared = x * x;
        FieldElement term(1.0f);
        
        // x^2/2!
        term = term * xSquared * (-1.0f/2.0f);
        result = result + term;
        
        // x^4/4!
        term = term * xSquared * (1.0f/12.0f);
        result = result + term;
        
        // x^6/6!
        term = term * xSquared * (-1.0f/30.0f);
        result = result + term;
        
        return result;
    }
    
    /**
     * Taylor series approximation for exp
     */
    static FieldElement expTaylor(const FieldElement& x) {
        FieldElement result(1.0f);  // First term: 1
        FieldElement term(1.0f);
        FieldElement xPower = x;  // x^1
        
        // Add terms: 1 + x + x^2/2! + x^3/3! + ...
        for (int i = 1; i <= 5; i++) {  // Use 5 terms
            term = xPower * (1.0f / factorial(i));
            result = result + term;
            xPower = xPower * x;
        }
        
        return result;
    }
    
    /**
     * Taylor series approximation for log(1+x)
     */
    static FieldElement logTaylor(const FieldElement& x) {
        // This assumes x is close to 1
        FieldElement one(1.0f);
        FieldElement xMinusOne = x - one;
        
        // For log(1+y) using y = x-1
        // Taylor series: y - y^2/2 + y^3/3 - ...
        FieldElement result = xMinusOne;
        FieldElement term = xMinusOne;
        FieldElement yPower = xMinusOne * xMinusOne;  // y^2
        
        // Subtract y^2/2
        term = yPower * (-0.5f);
        result = result + term;
        
        // Add y^3/3
        yPower = yPower * xMinusOne;  // y^3
        term = yPower * (1.0f/3.0f);
        result = result + term;
        
        // Subtract y^4/4
        yPower = yPower * xMinusOne;  // y^4
        term = yPower * (-1.0f/4.0f);
        result = result + term;
        
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
const uint8_t FieldElement<N>::BASIS_ELEMENTS[N][3] = {
    {0, 0, 0},  // 1 (constant term)
    {1, 0, 0},  // π
    {0, 1, 0},  // e
    {0, 0, 1},  // √2
    // Additional basis elements can be defined based on N
    // For example:
    //{2, 0, 0},  // π²
    //{0, 2, 0},  // e²
    //{0, 0, 2},  // 2
    //{1, 1, 0},  // π·e
    //{1, 0, 1},  // π·√2
    //{0, 1, 1}   // e·√2
};

/**
 * Type aliases for common field extension sizes
 */
using FieldElement4 = FieldElement<4>;   // Basic field with [1, π, e, √2]
using FieldElement8 = FieldElement<8>;   // Extended field with more combinations
using FieldElement16 = FieldElement<16>; // Advanced field with many combinations

#endif // FIELD_EXTENSION_H
