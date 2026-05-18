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

    std::cout << "Tests for version 3 passed!" << std::endl;
    return 0;
}
