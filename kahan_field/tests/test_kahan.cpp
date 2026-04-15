#include "../kahan_field.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <iomanip>

void test_kahan_scalar() {
    std::cout << "Testing Kahan scalar summation..." << std::endl;
    AlgebraicKahanSummator<double> kahan;

    kahan.add(1.0);
    // Use a value that is small but large enough to eventually make a difference
    // 1e-16 is below machine epsilon for 1.0 (which is ~2.22e-16)
    double tiny_val = 1e-16;
    int iterations = 10000;

    for (int i = 0; i < iterations; ++i) {
        kahan.add(tiny_val);
    }

    double naive = 1.0;
    for (int i = 0; i < iterations; ++i) {
        naive += tiny_val;
    }

    std::cout << "  Naive sum: " << std::fixed << std::setprecision(20) << naive << std::endl;
    std::cout << "  Kahan sum: " << kahan.sum() << std::endl;
    std::cout << "  Algebraic: " << kahan.algebraic_sum() << std::endl;
    std::cout << "  Expected:  " << (1.0 + iterations * tiny_val) << std::endl;

    assert(kahan.algebraic_sum() > 1.0);
    assert(naive == 1.0);

    std::cout << "Kahan scalar summation passed!" << std::endl;
}

void test_kahan_tensor() {
    std::cout << "Testing Kahan tensor summation..." << std::endl;
    Tensor<double> zero(2, 2, 0.0);
    AlgebraicKahanSummator<Tensor<double>> kahan(zero);

    Tensor<double> base({{1.0, 100.0}, {0.01, 1000.0}});
    Tensor<double> tiny({{1e-16, 1e-14}, {1e-18, 1e-13}});

    kahan.add(base);
    for (int i = 0; i < 1000; ++i) {
        kahan.add(tiny);
    }

    std::cout << "  Kahan tensor sum(0,0): " << kahan.algebraic_sum()(0,0) << std::endl;
    assert(kahan.algebraic_sum()(0,0) > 1.0);

    std::cout << "Kahan tensor summation passed!" << std::endl;
}

void test_cohomology() {
    std::cout << "Testing Cohomological analysis..." << std::endl;
    double a = 1.0;
    double b = 1e-16;
    double c = 1e-16;

    double defect = FloatingPointCohomology<double>::associativity_defect(a, b, c);
    std::cout << "  Associativity defect: " << defect << std::endl;

    std::cout << "Cohomological analysis completed." << std::endl;
}

int main() {
    test_kahan_scalar();
    test_kahan_tensor();
    test_cohomology();
    return 0;
}
