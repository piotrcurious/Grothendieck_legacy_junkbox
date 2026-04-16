#include <iostream>
#include <cassert>
#include <cmath>
#include <iomanip>
#include "../GaussianDualField.h"

void test_sin_cos() {
    using Field = GaussianDualField<double>;
    Field x(1.0, 0.2, 0.3, 0.01);

    Field sx = Field::sin(x);
    Field cx = Field::cos(x);
    Field tx = Field::tan(x);

    // sin^2 + cos^2 = 1?
    // In hyperbolic numbers, this is more complex: cosh^2 - sinh^2 = 1
    // Let's test sin/cos = tan
    Field t_check = sx / cx;

    std::cout << std::fixed << std::setprecision(10);
    std::cout << "Sin/Cos = Tan: nominal=" << t_check.nominal << ", tan=" << tx.nominal << std::endl;
    assert(std::abs(t_check.nominal - tx.nominal) < 1e-9);
    assert(std::abs(t_check.noise - tx.noise) < 1e-9);

    // Check derivatives
    // d/dx sin(x) = cos(x)
    Field x_der(1.0, 0, 1.0, 0.01);
    Field sin_der = Field::sin(x_der);
    Field cos_val = Field::cos(x_der);
    std::cout << "Sin' nominal=" << sin_der.delta << ", Cos nominal=" << cos_val.nominal << std::endl;
    assert(std::abs(sin_der.delta - cos_val.nominal) < 1e-9);

    std::cout << "Circular functions tests passed!" << std::endl;
}

int main() {
    test_sin_cos();
    return 0;
}
