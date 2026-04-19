#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "../test/Arduino.h"

// For not_strict.ino
#include "../not_strict.ino"

#include "test_data_generator.h"

void test_not_strict() {
    std::cout << "Testing BanachSpaceAnalyzer with noisy representative data..." << std::endl;
    banachAnalyzer.reset();

    // Generate complex trend with noise
    auto signal = banach::test::DataGenerator::generateComplexTrend(100, 0.1f);
    for (const auto& p : signal) {
        banachAnalyzer.addData(p.v, p.t);
    }
    banachAnalyzer.analyzeBanachSpace();

    // Rigorous assertions for Legendre projection
    std::cout << "Verifying Legendre projection for constant signal..." << std::endl;
    banachAnalyzer.reset();
    for(int i=0; i<10; ++i) banachAnalyzer.addData(5.0f);
    auto coeffs = banachAnalyzer.projectLegendre(3);
    // P0 should be approx 5.0, others approx 0
    assert(std::abs(coeffs[0] - 5.0f) < 1e-3);
    assert(std::abs(coeffs[1]) < 1e-3);

    std::cout << "Verifying Legendre projection for linear signal..." << std::endl;
    banachAnalyzer.reset();
    for(int i=0; i<11; ++i) banachAnalyzer.addData(static_cast<float>(i)); // t: 0..10, v: 0..10
    coeffs = banachAnalyzer.projectLegendre(3);
    // P1 (shifted x) should be non-zero
    assert(std::abs(coeffs[0] - 5.0f) < 1e-3); // Average is 5
    assert(std::abs(coeffs[1] - 5.0f) < 1e-3); // Range 0..10 maps to -1..1, slope 5*x matches 0..10

    std::cout << "BanachSpaceAnalyzer test finished." << std::endl;
}

int main() {
    test_not_strict();
    return 0;
}
