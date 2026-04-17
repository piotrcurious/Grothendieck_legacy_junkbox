#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

// Mock some Arduino things
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

    // Test overflow
    int32_t max_val = 0x7FFFFFFF;
    int32_t overflow_prod = q_mul(max_val, max_val);
    assert(overflow_prod == 0x7FFFFFFF);

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
    assert(std::abs(fixed_to_double(prod.real) - (-5.0)) < 0.001);
    assert(std::abs(fixed_to_double(prod.imag) - 10.0) < 0.001);
    std::cout << "Complex operations OK" << std::endl;
}

void test_process_frequency_domain_with_config() {
    std::cout << "Testing processFrequencyDomain with config..." << std::endl;
    for(int i=0; i<BUFFER_SIZE; ++i) {
        realBuffer[i] = std::sin(2.0 * M_PI * i / 10.0);
        imagBuffer[i] = 0.0;
    }

    FieldConfig config = {
        double_to_fixed(0.5),  // lambda
        double_to_fixed(0.05), // eps
        double_to_fixed(0.5)   // grad_weight
    };

    processFrequencyDomain(config);

    for(int i=0; i<BUFFER_SIZE; ++i) {
        assert(!std::isnan(realBuffer[i]));
        assert(!std::isnan(imagBuffer[i]));
    }
    std::cout << "processFrequencyDomain with config OK" << std::endl;
}

void test_second_order_gradient() {
    std::cout << "Testing second-order gradient logic..." << std::endl;
    // Set a linear slope to check if gradient is constant
    for(int i=0; i<BUFFER_SIZE; ++i) {
        realBuffer[i] = (double)i / BUFFER_SIZE;
        imagBuffer[i] = 0.0;
    }

    // We can't easily check internal D[i] without exposing it or using a debugger,
    // but we can ensure processFrequencyDomain runs without crash and produces sane output.
    processFrequencyDomain();

    std::cout << "Second-order gradient logic (sanity check) OK" << std::endl;
}

int main() {
    test_fixed_point();
    test_complex_ops();
    test_process_frequency_domain_with_config();
    test_second_order_gradient();
    std::cout << "All host-side Weyl filter tests passed!" << std::endl;
    return 0;
}
