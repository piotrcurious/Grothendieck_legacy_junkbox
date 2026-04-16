#include "../kahan_field.hpp"
#include <iostream>
#include <cassert>

void test_rational_basic() {
    std::cout << "Testing Rational field arithmetic..." << std::endl;
    Rational a(1, 2);
    Rational b(1, 3);

    assert(a + b == Rational(5, 6));
    assert(a - b == Rational(1, 6));
    assert(a * b == Rational(1, 6));
    assert(a / b == Rational(3, 2));

    Rational c(2, 4);
    assert(c == Rational(1, 2));

    std::cout << "Rational field arithmetic passed!" << std::endl;
}

void test_tensor_matmul_precision() {
    std::cout << "Testing compensated_matmul precision..." << std::endl;
    // We want a case where naive accumulation fails
    size_t N = 100;
    Tensor<double> A(1, N, 1.0);
    Tensor<double> B(N, 1, 1e-16);

    // dot product should be N * 1e-16
    Tensor<double> C_naive = A.matmul(B);
    Tensor<double> C_kahan = A.compensated_matmul(B);

    std::cout << "  Naive matmul result: " << C_naive(0, 0) << std::endl;
    std::cout << "  Kahan matmul result: " << C_kahan(0, 0) << std::endl;
    std::cout << "  Expected result:     " << (N * 1e-16) << std::endl;

    // Naive might lose precision depending on order, but with all 1e-16 it might actually work if N is small.
    // Let's try 1.0 + many tiny values
    size_t M = 10000;
    Tensor<double> A2(1, M + 1);
    A2(0, 0) = 1.0;
    for(size_t i=1; i<=M; ++i) A2(0, i) = 1e-16;

    Tensor<double> B2(M + 1, 1, 1.0);

    Tensor<double> C2_naive = A2.matmul(B2);
    Tensor<double> C2_kahan = A2.compensated_matmul(B2);

    std::cout << "  Naive matmul (1.0 + tiny): " << std::fixed << std::setprecision(20) << C2_naive(0, 0) << std::endl;
    std::cout << "  Kahan matmul (1.0 + tiny): " << C2_kahan(0, 0) << std::endl;

    assert(C2_kahan(0, 0) > 1.0);
    assert(C2_naive(0, 0) == 1.0);

    std::cout << "compensated_matmul precision passed!" << std::endl;
}

void test_scalar_tensor_ops() {
    std::cout << "Testing scalar-tensor operations..." << std::endl;
    Tensor<double> A({{1.0, 2.0}, {3.0, 4.0}});
    Tensor<double> B = A * 2.0;

    assert(B(0, 0) == 2.0);
    assert(B(1, 1) == 8.0);

    Tensor<double> C = B / 2.0;
    assert(C == A);

    std::cout << "scalar-tensor operations passed!" << std::endl;
}

int main() {
    test_rational_basic();
    test_tensor_matmul_precision();
    test_scalar_tensor_ops();
    return 0;
}
