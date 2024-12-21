#ifndef ALGEBRAIC_COMPUTE_H
#define ALGEBRAIC_COMPUTE_H

#include <vector>
#include <cmath>

// Polynomial Class
class Polynomial {
public:
    std::vector<int> coefficients; // Coefficients of the polynomial
    int modulus;                   // Modulus for modular arithmetic

    Polynomial(std::vector<int> coeffs, int mod);

    Polynomial reduceUsingGroebnerBasis(const std::vector<std::vector<int>>& groebnerBasis);
    Polynomial operator+(const Polynomial& other);
    Polynomial operator*(const Polynomial& other);
};

// Banach Space Class
class BanachSpace {
public:
    float calculateNorm(const Polynomial& poly, int p = 2); // Lp norm
    Polynomial regularize(const Polynomial& poly, float threshold);
};

// Automorphism Class
class Automorphism {
public:
    static Polynomial frobenius(const Polynomial& poly, int fieldOrder);
    static Polynomial bitwiseShift(const Polynomial& poly, int shiftAmount);
};

// Time Series Class
class TimeSeries {
public:
    std::vector<std::pair<long, float>> data; // Timestamp-value pairs

    Polynomial fitPolynomial(int degree);
    Polynomial reduceFeatures(const Polynomial& poly);
    float predictValue(long timestamp);
};

#endif
