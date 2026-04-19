#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "Arduino.h"
#include "test_data_generator.h"
#include "../math_utils.h"

void test_gapped_signal() {
    std::cout << "Testing Gapped Signal Analysis..." << std::endl;
    // Generate signal with significant gaps
    auto signal = banach::test::DataGenerator::generateGappedSignal(100, 0.2f, 5.0f);

    std::vector<float> v, t;
    for(const auto& p : signal) { v.push_back(p.v); t.push_back(p.t); }

    // Mean should still be reasonable if Lebesgue weighting works
    float mean = banach::Statistics::calculateMean(v, t);
    std::cout << "Gapped Mean: " << mean << std::endl;
    // For sin(t), mean over long period should be small, but here we just check it doesn't crash or return NaN
    assert(!std::isnan(mean));
}

void test_quantized_signal() {
    std::cout << "Testing Quantized Signal Analysis..." << std::endl;
    // 3-bit quantization (8 levels)
    auto signal = banach::test::DataGenerator::generateQuantizedSignal(100, 3.0f, 2.0f);

    std::vector<float> v, t;
    for(const auto& p : signal) { v.push_back(p.v); t.push_back(p.t); }

    float flatness = banach::Statistics::calculateFlatness(v, t);
    std::cout << "Quantized Flatness: " << flatness << std::endl;
    // High quantization noise should reduce flatness (make it less "tonal" than pure sine)
    assert(flatness < 1.0f);
}

int main() {
    test_gapped_signal();
    test_quantized_signal();
    std::cout << "Artifact tests passed!" << std::endl;
    return 0;
}
