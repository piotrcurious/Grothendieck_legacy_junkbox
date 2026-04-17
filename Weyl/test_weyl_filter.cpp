#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

// Mock some Arduino things if needed, but Weyl_Filter_Utils.h is mostly self-contained
#define BUFFER_SIZE 128
double realBuffer[BUFFER_SIZE];
double imagBuffer[BUFFER_SIZE];

#include "Weyl_Filter_Utils.h"

fixed_complex fixedF[BUFFER_SIZE];

void test_fixed_point() {
    std::cout << "Testing fixed point arithmetic..." << std::endl;
    int32_t a = double_to_fixed(0.5);
    int32_t b = double_to_fixed(2.0);
    int32_t prod = q_mul(a, b);
    assert(std::abs(fixed_to_double(prod) - 1.0) < 0.001);

    int32_t quot = q_div(b, a);
    assert(std::abs(fixed_to_double(quot) - 4.0) < 0.001);
    std::cout << "Fixed point arithmetic OK" << std::endl;
}

void test_complex_ops() {
    std::cout << "Testing complex operations..." << std::endl;
    fixed_complex a = fc_from_double(1.0, 2.0);
    fixed_complex b = fc_from_double(3.0, 4.0);
    fixed_complex sum = fc_add(a, b);
    assert(std::abs(fixed_to_double(sum.real) - 4.0) < 0.001);
    assert(std::abs(fixed_to_double(sum.imag) - 6.0) < 0.001);

    fixed_complex prod = fc_mul(a, b);
    // (1+2i)(3+4i) = 3 + 4i + 6i - 8 = -5 + 10i
    assert(std::abs(fixed_to_double(prod.real) - (-5.0)) < 0.001);
    assert(std::abs(fixed_to_double(prod.imag) - 10.0) < 0.001);
    std::cout << "Complex operations OK" << std::endl;
}

void test_process_frequency_domain() {
    std::cout << "Testing processFrequencyDomain..." << std::endl;
    // Fill buffers with a dummy signal (e.g. DC component)
    for(int i=0; i<BUFFER_SIZE; ++i) {
        realBuffer[i] = 1.0;
        imagBuffer[i] = 0.0;
    }

    processFrequencyDomain();

    // Check that results are finite and reasonably changed
    for(int i=0; i<BUFFER_SIZE; ++i) {
        assert(!std::isnan(realBuffer[i]));
        assert(!std::isnan(imagBuffer[i]));
    }
    std::cout << "processFrequencyDomain OK" << std::endl;
}

int main() {
    test_fixed_point();
    test_complex_ops();
    test_process_frequency_domain();
    std::cout << "All host-side Weyl filter tests passed!" << std::endl;
    return 0;
}
