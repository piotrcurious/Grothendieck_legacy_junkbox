#include "../kahan_field.hpp"
#include <iostream>
#include <cassert>

void test_tensor_arithmetic() {
    std::cout << "Testing Tensor arithmetic..." << std::endl;
    Tensor<double> A({{1.0, 2.0}, {3.0, 4.0}});
    Tensor<double> B({{5.0, 6.0}, {7.0, 8.0}});

    Tensor<double> C = A + B;
    assert(C(0, 0) == 6.0);
    assert(C(1, 1) == 12.0);

    Tensor<double> D = A * B; // Element-wise
    assert(D(0, 0) == 5.0);
    assert(D(1, 1) == 32.0);

    std::cout << "Tensor arithmetic passed!" << std::endl;
}

void test_tensor_gfp() {
    std::cout << "Testing Tensor over GF_p..." << std::endl;
    Tensor<GF_p<5>> A({{GF_p<5>(1), GF_p<5>(2)}, {GF_p<5>(3), GF_p<5>(4)}});
    Tensor<GF_p<5>> B({{GF_p<5>(4), GF_p<5>(3)}, {GF_p<5>(2), GF_p<5>(1)}});

    Tensor<GF_p<5>> C = A + B;
    assert(C(0, 0) == GF_p<5>(0));
    assert(C(1, 1) == GF_p<5>(0));

    std::cout << "Tensor over GF_p passed!" << std::endl;
}

int main() {
    test_tensor_arithmetic();
    test_tensor_gfp();
    return 0;
}
