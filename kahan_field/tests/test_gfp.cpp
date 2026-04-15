#include "../kahan_field.hpp"
#include <iostream>
#include <cassert>

void test_gfp_basic() {
    std::cout << "Testing GF_p basic operations..." << std::endl;
    GF_p<7> a(3);
    GF_p<7> b(5);

    assert(a + b == GF_p<7>(1)); // (3 + 5) % 7 = 1
    assert(a - b == GF_p<7>(5)); // (3 - 5 + 7) % 7 = 5
    assert(a * b == GF_p<7>(1)); // (3 * 5) % 7 = 1
    assert(a / b == GF_p<7>(2)); // 3 * inv(5) = 3 * 3 = 9 % 7 = 2

    std::cout << "GF_p basic operations passed!" << std::endl;
}

void test_gfp_inverse() {
    std::cout << "Testing GF_p modular inverse..." << std::endl;
    for (int i = 1; i < 13; ++i) {
        GF_p<13> a(i);
        GF_p<13> inv_a = GF_p<13>(1) / a;
        assert(a * inv_a == GF_p<13>(1));
    }

    try {
        GF_p<13> zero(0);
        GF_p<13> fail = GF_p<13>(1) / zero;
        assert(false && "Should have thrown division by zero");
    } catch (const std::runtime_error& e) {
        std::cout << "Caught expected error: " << e.what() << std::endl;
    }

    std::cout << "GF_p modular inverse passed!" << std::endl;
}

int main() {
    test_gfp_basic();
    test_gfp_inverse();
    return 0;
}
