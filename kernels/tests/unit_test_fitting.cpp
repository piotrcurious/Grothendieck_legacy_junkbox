#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "../fitting_utils.h"

#define ASSERT_NEAR(a, b, epsilon) \
    if (std::abs((a) - (b)) > (epsilon)) { \
        std::cerr << "Assertion failed: " << #a << " (" << (a) << ") != " << #b << " (" << (b) << ") within " << #epsilon << std::endl; \
        exit(1); \
    }

void test_linear_fit() {
    std::cout << "Running test_linear_fit..." << std::endl;
    float x[5] = {0, 1, 2, 3, 4};
    float y[5] = {10, 12, 14, 16, 18}; // y = 2x + 10
    float m, b;
    linearFit(x, y, 5, m, b);
    ASSERT_NEAR(m, 2.0, 1e-5);
    ASSERT_NEAR(b, 10.0, 1e-5);
    std::cout << "test_linear_fit PASSED" << std::endl;
}

void test_exponential_fit() {
    std::cout << "Running test_exponential_fit..." << std::endl;
    float x[5] = {0, 1, 2, 3, 4};
    float a_true = 5.0;
    float b_true = 0.1;
    float y[5];
    for(int i=0; i<5; i++) y[i] = a_true * exp(b_true * x[i]);

    float a_fit, b_fit;
    exponentialFitFredholm(x, y, 5, a_fit, b_fit);

    // Note: Integral-based fit might have small bias for very few points,
    // but should be close.
    ASSERT_NEAR(b_fit, b_true, 0.01);
    ASSERT_NEAR(a_fit, a_true, 0.1);
    std::cout << "test_exponential_fit PASSED" << std::endl;
}

void test_exponential_fit_offset() {
    std::cout << "Running test_exponential_fit_offset..." << std::endl;
    float x[5] = {10, 11, 12, 13, 14}; // Non-zero start
    float a_true = 2.0;
    float b_true = 0.05;
    float y[5];
    for(int i=0; i<5; i++) y[i] = a_true * exp(b_true * x[i]);

    float a_fit, b_fit;
    exponentialFitFredholm(x, y, 5, a_fit, b_fit);

    ASSERT_NEAR(b_fit, b_true, 0.01);
    ASSERT_NEAR(a_fit, a_true, 0.1);
    std::cout << "test_exponential_fit_offset PASSED" << std::endl;
}

void test_polynomial_fit() {
    std::cout << "Running test_polynomial_fit..." << std::endl;
    float x[5] = {0, 1, 2, 3, 4};
    float y[5] = {1, 3, 7, 13, 21}; // y = x^2 + x + 1
    float coeffs[3];
    polynomialFit(x, y, 5, 2, coeffs);
    ASSERT_NEAR(coeffs[0], 1.0, 1e-4);
    ASSERT_NEAR(coeffs[1], 1.0, 1e-4);
    ASSERT_NEAR(coeffs[2], 1.0, 1e-4);
    std::cout << "test_polynomial_fit PASSED" << std::endl;
}

int main() {
    test_linear_fit();
    test_exponential_fit();
    test_exponential_fit_offset();
    test_polynomial_fit();
    std::cout << "\nALL UNIT TESTS PASSED!" << std::endl;
    return 0;
}
