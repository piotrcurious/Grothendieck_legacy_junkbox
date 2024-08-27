#ifndef PRIMITIVE_POLYNOMIAL_H
#define PRIMITIVE_POLYNOMIAL_H

#include <vector>
#include <iostream>
#include <cmath>
#include <bitset>

class PrimitivePolynomial {
public:
    PrimitivePolynomial(int degree);

    bool isPrimitive(const std::vector<int>& poly);
    std::vector<std::vector<int>> findPrimitivePolynomials();
    void printPolynomial(const std::vector<int>& poly);

private:
    int degree;
    int order;

    bool isIrreducible(const std::vector<int>& poly);
    int calculatePeriod(const std::vector<int>& poly);
    bool dividePolynomials(const std::vector<int>& dividend, const std::vector<int>& divisor);

    std::vector<std::vector<int>> generateAllPolynomials(int deg);
};

#endif
