#include <iostream>
#include <cassert>
#include <cmath>
#include "../GaussianDualField.h"

void test_basic_arithmetic() {
    using Field = GaussianDualField<double>;
    Field x(1.0, 0.2, 0.3, 0.01);
    Field y(0.5, 0.1, 0.05, 0.01);

    // Addition
    Field sum = x + y;
    assert(std::abs(sum.nominal - 1.5) < 1e-9);
    assert(std::abs(sum.noise - 0.3) < 1e-9);
    assert(std::abs(sum.delta - 0.35) < 1e-9);

    // Subtraction
    Field diff = x - y;
    assert(std::abs(diff.nominal - 0.5) < 1e-9);
    assert(std::abs(diff.noise - 0.1) < 1e-9);
    assert(std::abs(diff.delta - 0.25) < 1e-9);

    // Multiplication
    Field prod = x * y;
    assert(std::abs(prod.nominal - 0.5002) < 1e-9);
    assert(std::abs(prod.noise - 0.2) < 1e-9);
    assert(std::abs(prod.delta - 0.2) < 1e-9);

    std::cout << "Basic arithmetic tests passed!" << std::endl;
}

void test_division() {
    using Field = GaussianDualField<double>;
    Field x(1.0, 0.2, 0.3, 0.01);
    Field y(0.5, 0.1, 0.05, 0.01);

    Field z = x / y;
    Field check = z * y;

    std::cout << "Division result: nominal=" << z.nominal << ", noise=" << z.noise << ", delta=" << z.delta << std::endl;
    std::cout << "Check result: nominal=" << check.nominal << ", noise=" << check.noise << ", delta=" << check.delta << std::endl;

    assert(std::abs(check.nominal - x.nominal) < 1e-9);
    assert(std::abs(check.noise - x.noise) < 1e-9);
    assert(std::abs(check.delta - x.delta) < 1e-3);

    std::cout << "Division test passed!" << std::endl;
}

int main() {
    test_basic_arithmetic();
    test_division();
    return 0;
}
