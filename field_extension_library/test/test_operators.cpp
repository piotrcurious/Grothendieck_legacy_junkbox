#include "Arduino.h"
#include "../FieldExtension.h"
#include "../2/FieldExtension.h"
#include "../3/FieldExtension.h"
#include <cassert>
#include <iostream>

void testV1InPlace();
void testV2Transcendental();
void testV3Arithmetic();
void testScalarLeft();

int main() {
    std::cout << "Testing In-place Operators and Cross-version Compatibility..." << std::endl;

    testV1InPlace();
    testV2Transcendental();
    testV3Arithmetic();
    testScalarLeft();

    std::cout << "All cross-version and operator tests completed successfully!" << std::endl;
    return 0;
}

void testV1InPlace() {
    std::cout << "\n=== Testing V1 In-place Operators ===" << std::endl;
    FieldElement4 a(1.0f);
    a += FieldElement4::pi();
    std::cout << "1 + pi = " << a.toFloat() << std::endl;
    assert(std::abs(a.toFloat() - (1.0f + M_PI)) < 1e-6);

    a *= 2.0f;
    std::cout << "(1 + pi) * 2 = " << a.toFloat() << std::endl;
    assert(std::abs(a.toFloat() - 2.0f * (1.0f + M_PI)) < 1e-5);
}

void testV2Transcendental() {
    std::cout << "\n=== Testing V2 Transcendental Functions ===" << std::endl;
    using ExtPi2 = FieldExt<float, 3.1415926535f, 2>;
    ExtPi2 x(0.0f);
    x[1] = 1.0f; // pi

    ExtPi2 s = sin(x);
    std::cout << "sin(pi) [v2] = " << s.eval() << std::endl;
    assert(std::abs(s.eval()) < 1e-6);

    ExtPi2 c = cos(x);
    std::cout << "cos(pi) [v2] = " << c.eval() << std::endl;
    assert(std::abs(c.eval() + 1.0f) < 1e-6);
}

void testV3Arithmetic() {
    std::cout << "\n=== Testing V3 In-place and Scalar Arithmetic ===" << std::endl;
    FieldExtension fe(1.0f);
    fe += FieldExtension({{1.0f, Basis::Pi}});
    std::cout << "1 + pi [v3] = " << fe.evaluate() << std::endl;
    assert(std::abs(fe.evaluate() - (1.0f + M_PI)) < 1e-6);

    fe *= 0.5f;
    std::cout << "(1 + pi)/2 [v3] = " << fe.evaluate() << std::endl;
    assert(std::abs(fe.evaluate() - (1.0f + M_PI) * 0.5f) < 1e-6);
}

void testScalarLeft() {
    std::cout << "\n=== Testing Scalar-Left Arithmetic ===" << std::endl;
    FieldElement4 a = FieldElement4::pi();
    FieldElement4 b = 2.0f * a;
    std::cout << "2 * pi = " << b.toFloat() << std::endl;
    assert(std::abs(b.toFloat() - 2.0f * M_PI) < 1e-6);

    FieldElement4 c = 1.0f + a;
    std::cout << "1 + pi = " << c.toFloat() << std::endl;
    assert(std::abs(c.toFloat() - (1.0f + M_PI)) < 1e-6);
}
