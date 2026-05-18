#include "Arduino.h"
#include "../2/FieldExtension.h"
#include <iostream>
#include <cassert>

int main() {
    std::cout << "Testing field_extension_library/2/FieldExtension.h..." << std::endl;

    // In C++20 we can use float as NTTP
    using ExtPi2 = FieldExt<float, 3.1415926535f, 2>;

    ExtPi2 a(1.0f); // 1 + 0*pi
    ExtPi2 b; b[1] = 1.0f; // 0 + 1*pi

    ExtPi2 sum = a + b;
    std::cout << "1 + pi = " << sum.eval() << std::endl;
    assert(std::abs(sum.eval() - 4.14159265f) < 1e-5);

    ExtPi2 prod = b * b; // pi * pi = pi^2. In degree 2 it's truncated.
    std::cout << "pi * pi (truncated) = " << prod.eval() << std::endl;
    assert(prod[0] == 0 && prod[1] == 0);

    ExtPi2 c(2.0f);
    ExtPi2 inv_c = c.recip();
    std::cout << "recip(2) = " << inv_c.eval() << std::endl;
    assert(std::abs(inv_c.eval() - 0.5f) < 1e-5);

    std::cout << "Tests for version 2 passed!" << std::endl;
    return 0;
}
