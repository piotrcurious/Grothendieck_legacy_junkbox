#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "../test/Arduino.h"

#include "../3_space.ino"

void test_uneven_sampling() {
    std::cout << "Testing Lebesgue-style integration for uneven sampling (3_space.ino)..." << std::endl;

    BanachSpace<float, 1> space;

    // Constant signal 1.0, unevenly sampled
    // t: 0, 1, 10
    // v: 1, 1, 1
    // Total integral = 10, Mean = 1, L2 norm should be sqrt(1^2) = 1
    space.addDataPoint({1.0}, 0.0);
    space.addDataPoint({1.0}, 1.0);
    space.addDataPoint({1.0}, 10.0);

    float l2 = space.computeLpNorm(2);
    std::cout << "L2 norm (expected ~1.0): " << l2 << std::endl;
    assert(std::abs(l2 - 1.0) < 1e-3);

    // Changing signal
    space.reset();
    space.addDataPoint({0.0}, 0.0);
    space.addDataPoint({10.0}, 1.0); // Large spike but only for a short time
    space.addDataPoint({0.0}, 10.0);

    // Lebesgue integral of |f|^2 over [0, 10]:
    // [0, 1]: avg val is 5, contribution 25 * 1 = 25
    // [1, 10]: avg val is 5, contribution 25 * 9 = 225
    // Total = 250. Weighted norm = sqrt(250 / 10) = sqrt(25) = 5
    l2 = space.computeLpNorm(2);
    std::cout << "L2 norm for triangle (expected ~5.0): " << l2 << std::endl;
    assert(std::abs(l2 - 5.0) < 1e-3);

    std::cout << "uneven_sampling test finished." << std::endl;
}

int main() {
    test_uneven_sampling();
    return 0;
}
