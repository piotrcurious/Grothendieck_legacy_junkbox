#include "Arduino.h"
#include "../FieldExtension.h"
#include <cassert>
#include <iostream>
#include <iomanip>

void testFieldElement8();
void testFieldElement16();
void testExtendedPrecision();
void testRangeReduction();
void testIdentity();
void testNewFunctions();

int main() {
    std::cout << std::fixed << std::setprecision(10);
    std::cout << "Starting Expanded ESP32 Field Extension Library Tests..." << std::endl;

    testFieldElement8();
    testFieldElement16();
    testExtendedPrecision();
    testRangeReduction();
    testIdentity();
    testNewFunctions();

    std::cout << "All expanded tests completed successfully!" << std::endl;
    return 0;
}

void testFieldElement8() {
    std::cout << "\n=== Testing FieldElement8 ===" << std::endl;
    FieldElement8 pi = FieldElement8::pi();
    FieldElement8 e = FieldElement8::e();

    // pi*e should be term 6
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

    // pi*e*sqrt2 should be term 15
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
    // pi^2 is term 4
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
    std::cout << "\n=== Testing New Functions (sqrt, atan, asin, acos) ===" << std::endl;
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
