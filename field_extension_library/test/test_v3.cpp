#include "Arduino.h"
#include "../3/FieldExtension.h"
#include <iostream>
#include <cassert>

int main() {
    std::cout << "Testing field_extension_library/3/FieldExtension.h..." << std::endl;

    FieldExtension fe1({{1.0f, Basis::One}, {1.0f, Basis::Pi}});
    std::cout << "fe1 (1 + pi) evaluation: " << fe1.evaluate() << std::endl;
    assert(std::abs(fe1.evaluate() - 4.14159265f) < 1e-5);

    FieldExtension fe2({{1.0f, Basis::E}});
    FieldExtension fe3 = fe1 + fe2;
    std::cout << "fe3 (1 + pi + e) evaluation: " << fe3.evaluate() << std::endl;
    assert(std::abs(fe3.evaluate() - (1.0f + 3.14159265f + 2.71828182f)) < 1e-5);

    FieldExtension fe4 = fe2 * 2.0f;
    std::cout << "fe4 (2*e) evaluation: " << fe4.evaluate() << std::endl;
    assert(std::abs(fe4.evaluate() - 2.0f * 2.71828182f) < 1e-5);

    // Test Pi * Pi improvement
    FieldExtension pi({{1.0f, Basis::Pi}});
    FieldExtension pi2 = pi * pi;
    std::cout << "pi^2 evaluation: " << pi2.evaluate() << " (should be ~9.869)" << std::endl;
    assert(std::abs(pi2.evaluate() - 9.8696044f) < 1e-4);

    // Test division
    FieldExtension two(2.0f);
    FieldExtension half = FieldExtension(1.0f) / two;
    std::cout << "1/2 evaluation: " << half.evaluate() << std::endl;
    assert(std::abs(half.evaluate() - 0.5f) < 1e-6);

    std::cout << "Tests for version 3 passed!" << std::endl;
    return 0;
}
