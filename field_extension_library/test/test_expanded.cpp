#include "Arduino.h"
#include "../FieldExtension.h"
#include <cassert>
#include <iostream>
#include <iomanip>
#include <chrono>

void testFieldElement8();
void testFieldElement16();
void testExtendedPrecision();
void testRangeReduction();
void testIdentity();
void testNewFunctions();
void testPow();
void testHyperbolic();
void testMultiplicationBenchmark();

int main() {
    std::cout << std::fixed << std::setprecision(10);
    std::cout << "Starting Expanded ESP32 Field Extension Library Tests..." << std::endl;

    testFieldElement8();
    testFieldElement16();
    testExtendedPrecision();
    testRangeReduction();
    testIdentity();
    testNewFunctions();
    testPow();
    testHyperbolic();
    testMultiplicationBenchmark();

    std::cout << "All expanded tests completed successfully!" << std::endl;
    return 0;
}

void testFieldElement8() {
    std::cout << "\n=== Testing FieldElement8 ===" << std::endl;
    FieldElement8 pi = FieldElement8::pi();
    FieldElement8 e = FieldElement8::e();

    FieldElement8 pie = pi * e;
    std::cout << "pi * e = " << pie.toFloat() << std::endl;
    std::cout << "Term 6 coefficient (pi*e): " << pie.getCoefficient(6) << std::endl;
    assert(std::abs(pie.getCoefficient(6) - 1.0f) < 1e-6);
}

void testFieldElement16() {
    std::cout << "\n=== Testing FieldElement16 ===" << std::endl;
    FieldElement16 pi = FieldElement16::pi();
    FieldElement16 e = FieldElement16::e();
    FieldElement16 sqrt2 = FieldElement16::sqrt2();

    FieldElement16 triple = pi * e * sqrt2;
    std::cout << "pi * e * sqrt2 = " << triple.toFloat() << std::endl;
    std::cout << "Term 15 coefficient (pi*e*sqrt2): " << triple.getCoefficient(15) << std::endl;
    assert(std::abs(triple.getCoefficient(15) - 1.0f) < 1e-6);
}

void testExtendedPrecision() {
    std::cout << "\n=== Testing Extended Precision (pi^2) ===" << std::endl;
    FieldElement8 pi = FieldElement8::pi();
    FieldElement8 pi2 = pi * pi;

    std::cout << "pi^2 = " << pi2.toFloat() << std::endl;
    std::cout << "Term 4 coefficient (pi^2): " << pi2.getCoefficient(4) << std::endl;
    assert(std::abs(pi2.getCoefficient(4) - 1.0f) < 1e-6);
}

void testRangeReduction() {
    std::cout << "\n=== Testing Range Reduction ===" << std::endl;
    FieldElement4 pi = FieldElement4::pi();
    FieldElement4 twoPi = pi * 2.0f;

    std::cout << "sin(2.5 * pi) = " << sin(pi * 2.5f).toFloat() << " (should be 1)" << std::endl;
    assert(std::abs(sin(pi * 2.5f).toFloat() - 1.0f) < 1e-6);

    std::cout << "cos(2 * pi) = " << cos(twoPi).toFloat() << " (should be 1)" << std::endl;
    assert(std::abs(cos(twoPi).toFloat() - 1.0f) < 1e-6);

    std::cout << "sin(pi) = " << sin(pi).toFloat() << " (should be 0)" << std::endl;
    assert(std::abs(sin(pi).toFloat()) < 1e-6);
}

void testIdentity() {
    std::cout << "\n=== Testing Identity (pi+e)^2 - pi^2 - e^2 - 2*pi*e ===" << std::endl;
    FieldElement8 pi = FieldElement8::pi();
    FieldElement8 e = FieldElement8::e();
    FieldElement8 sum = pi + e;
    FieldElement8 squared = sum * sum;
    FieldElement8 pi_squared = pi * pi;
    FieldElement8 e_squared = e * e;
    FieldElement8 two_pi_e = pi * e * 2.0f;
    FieldElement8 result = squared - pi_squared - e_squared - two_pi_e;

    std::cout << "Result evaluation: " << result.toFloat() << " (should be 0)" << std::endl;
    assert(std::abs(result.toFloat()) < 1e-6);
    for(int i=0; i<8; i++) {
        assert(std::abs(result.getCoefficient(i)) < 1e-6);
    }
}

void testNewFunctions() {
    std::cout << "\n=== Testing New Functions (sqrt, atan, asin, acos, atan2) ===" << std::endl;
    FieldElement4 pi = FieldElement4::pi();
    FieldElement4 pi2 = pi * pi;

    FieldElement4 s_pi2 = sqrt(pi2);
    std::cout << "sqrt(pi^2) = " << s_pi2.toFloat() << " (should be pi)" << std::endl;
    assert(std::abs(s_pi2.toFloat() - M_PI) < 1e-5);

    FieldElement4 one(1.0f);
    FieldElement4 a_one = atan(one);
    std::cout << "atan(1) = " << a_one.toFloat() << " (should be pi/4)" << std::endl;
    assert(std::abs(a_one.toFloat() - M_PI * 0.25f) < 1e-5);

    FieldElement4 zero(0.0f);
    FieldElement4 as_zero = asin(zero);
    std::cout << "asin(0) = " << as_zero.toFloat() << " (should be 0)" << std::endl;
    assert(std::abs(as_zero.toFloat()) < 1e-6);

    FieldElement4 ac_zero = acos(zero);
    std::cout << "acos(0) = " << ac_zero.toFloat() << " (should be pi/2)" << std::endl;
    assert(std::abs(ac_zero.toFloat() - M_PI * 0.5f) < 1e-5);

    FieldElement4 y(1.0f), x(1.0f);
    FieldElement4 a2 = atan2(y, x);
    std::cout << "atan2(1, 1) = " << a2.toFloat() << " (should be pi/4)" << std::endl;
    assert(std::abs(a2.toFloat() - M_PI * 0.25f) < 1e-5);
}

void testPow() {
    std::cout << "\n=== Testing Pow ===" << std::endl;
    FieldElement4 pi = FieldElement4::pi();
    FieldElement4 p2 = pow(pi, 2.0f);
    std::cout << "pow(pi, 2) = " << p2.toFloat() << " (should be pi^2)" << std::endl;
    assert(std::abs(p2.toFloat() - M_PI * M_PI) < 1e-4);

    FieldElement4 e = FieldElement4::e();
    FieldElement4 pe = pow(e, pi);
    std::cout << "pow(e, pi) = " << pe.toFloat() << " (should be e^pi)" << std::endl;
    assert(std::abs(pe.toFloat() - std::pow(M_E, M_PI)) < 1e-4);
}

void testHyperbolic() {
    std::cout << "\n=== Testing Hyperbolic Functions ===" << std::endl;
    FieldElement4 x(0.5f);
    FieldElement4 s = sinh(x);
    FieldElement4 c = cosh(x);
    // cosh^2 - sinh^2 = 1
    FieldElement4 res = c * c - s * s;
    std::cout << "cosh(0.5)^2 - sinh(0.5)^2 = " << res.toFloat() << " (should be 1)" << std::endl;
    assert(std::abs(res.toFloat() - 1.0f) < 1e-6);

    FieldElement4 as = asinh(s);
    std::cout << "asinh(sinh(0.5)) = " << as.toFloat() << " (should be 0.5)" << std::endl;
    assert(std::abs(as.toFloat() - 0.5f) < 1e-6);
}

void testMultiplicationBenchmark() {
    std::cout << "\n=== Testing Multiplication Benchmark ===" << std::endl;
    FieldElement16 a;
    for(int i=0; i<16; i++) a.setCoefficient(i, (float)i/10.0f);
    FieldElement16 b = a;

    auto start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<10000; i++) {
        a = a * b;
        // Keep coefficients small to avoid overflow during benchmark
        if (a.norm() > 10.0f) a = a * 0.1f;
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "10000 multiplications (FieldElement16) took: " << diff.count() << "s" << std::endl;
}
