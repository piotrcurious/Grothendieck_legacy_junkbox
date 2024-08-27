#include "PrimitivePolynomial.h"

// Constructor
PrimitivePolynomial::PrimitivePolynomial(int degree) : degree(degree) {
    order = (1 << degree) - 1; // 2^degree - 1
}

// Check if polynomial is primitive
bool PrimitivePolynomial::isPrimitive(const std::vector<int>& poly) {
    if (!isIrreducible(poly))
        return false;

    return calculatePeriod(poly) == order;
}

// Check if polynomial is irreducible
bool PrimitivePolynomial::isIrreducible(const std::vector<int>& poly) {
    int polyDegree = poly.size() - 1;
    for (int i = 1; i <= polyDegree / 2; i++) {
        if ((order % ((1 << i) - 1)) == 0) {
            std::vector<std::vector<int>> testPolys = generateAllPolynomials(i);
            for (const auto& divisor : testPolys) {
                if (dividePolynomials(poly, divisor)) {
                    return false; // Reducible if divisible by any non-trivial polynomial
                }
            }
        }
    }
    return true; // Irreducible if no divisors found
}

// Polynomial division logic to check divisibility
bool PrimitivePolynomial::dividePolynomials(const std::vector<int>& dividend, const std::vector<int>& divisor) {
    std::vector<int> remainder = dividend;

    int divisorDegree = divisor.size() - 1;
    int remainderDegree = remainder.size() - 1;

    while (remainderDegree >= divisorDegree) {
        if (remainder[remainderDegree] == 1) {
            for (int i = 0; i <= divisorDegree; ++i) {
                remainder[remainderDegree - i] ^= divisor[divisorDegree - i];
            }
        }
        remainderDegree--;
    }

    // If remainder is zero, dividend is divisible by divisor
    for (int bit : remainder) {
        if (bit != 0) return false;
    }
    return true;
}

// Calculate period of the polynomial
int PrimitivePolynomial::calculatePeriod(const std::vector<int>& poly) {
    std::vector<int> state(poly.size(), 0);
    state[0] = 1;
    int period = 0;
    do {
        int feedback = 0;
        for (size_t i = 0; i < poly.size(); ++i) {
            feedback ^= state[i] & poly[i];
        }

        for (size_t i = state.size() - 1; i > 0; --i) {
            state[i] = state[i - 1];
        }
        state[0] = feedback;

        ++period;
    } while (state != std::vector<int>(poly.size(), 0));
    
    return period;
}

// Generate all polynomials of a certain degree
std::vector<std::vector<int>> PrimitivePolynomial::generateAllPolynomials(int deg) {
    std::vector<std::vector<int>> result;
    int numPolys = 1 << deg;
    for (int i = 1; i < numPolys; ++i) { // Start from 1 to skip the zero polynomial
        std::vector<int> poly(deg + 1);
        for (int j = 0; j < deg; ++j) {
            poly[j] = (i >> j) & 1;
        }
        poly[deg] = 1; // Ensure the polynomial is of degree `deg`
        result.push_back(poly);
    }
    return result;
}

// Find all primitive polynomials
std::vector<std::vector<int>> PrimitivePolynomial::findPrimitivePolynomials() {
    std::vector<std::vector<int>> primitives;
    int numPolys = 1 << (degree - 1);

    for (int i = 1; i < numPolys; ++i) {
        std::vector<int> poly(degree);
        for (int j = 0; j < degree; ++j) {
            poly[j] = (i >> j) & 1;
        }
        poly.push_back(1); // Always end with x^degree

        if (isPrimitive(poly)) {
            primitives.push_back(poly);
        }
    }
    return primitives;
}

// Print polynomial
void PrimitivePolynomial::printPolynomial(const std::vector<int>& poly) {
    bool first = true;
    for (int i = poly.size() - 1; i >= 0; --i) {
        if (poly[i]) {
            if (!first) std::cout << " + ";
            if (i > 0) std::cout << "x^" << i;
            else std::cout << "1";
            first = false;
        }
    }
    std::cout << std::endl;
}
