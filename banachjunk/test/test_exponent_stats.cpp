#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "../test/Arduino.h"
#include "test_data_generator.h"

// For exponent_stats.ino
#include "../exponent_stats.ino"

void test_exponent_stats() {
    std::cout << "Testing ExponentialDetector with representative data..." << std::endl;

    // Generate clean exponential growth: y = exp(0.5 * t)
    for (int i = 0; i < 20; ++i) {
        expDetector.addDataPoint(std::exp(0.5f * i * 0.1f));
    }

    auto expCharacteristics = expDetector.analyzeExponentialProperties();
    std::cout << "Base Exponent: " << expCharacteristics.baseExponent << std::endl;
    std::cout << "Growth Rate: " << expCharacteristics.growthRate << std::endl;
    std::cout << "R-Squared: " << expCharacteristics.rSquared << std::endl;

    std::cout << "\nTesting StatisticalBanachSpace..." << std::endl;
    std::vector<std::vector<float>> statisticalMultiData = {
        {1.2, 2.4, 4.8, 9.6, 19.2},   // Dimension 1
        {3.5, 7.0, 14.0, 28.0, 56.0},  // Dimension 2
        {2.1, 4.2, 8.4, 16.8, 33.6}    // Dimension 3
    };

    for (size_t i = 0; i < statisticalMultiData[0].size(); ++i) {
        std::vector<float> point = {
            statisticalMultiData[0][i],
            statisticalMultiData[1][i],
            statisticalMultiData[2][i]
        };
        statisticalSpace.addStatisticalDataPoint(point);
    }

    std::cout << "\nTesting StatisticalBanachSpace with complex stochastic data..." << std::endl;
    statisticalSpace.reset();

    // Use Generator for a Random Walk (High Hurst, low entropy if drift is clear)
    auto rw = banach::test::DataGenerator::generateRandomWalk(100, 0.1f, 0.5f);
    for(const auto& p : rw) {
        statisticalSpace.addStatisticalDataPoint({p.v, p.v * 1.5f, std::sin(p.t)}, p.t);
    }

    std::cout << "Analysis for Random Walk + Drift + Sine segment:" << std::endl;
    statisticalSpace.performStatisticalAnalysis();

    // Specific Hurst check for Random Walk (expect > 0.5)
    // and ApEn check for Sine vs Noise
    statisticalSpace.reset();
    auto noisySine = banach::test::DataGenerator::generateNoisySine(100, 1.0f, 1.0f, 0.05f);
    for(const auto& p : noisySine) statisticalSpace.addStatisticalDataPoint({p.v, p.v, p.v}, p.t);

    std::cout << "\nAnalysis for Noisy Sine Wave (N=100):" << std::endl;
    statisticalSpace.performStatisticalAnalysis();

    // Hurst Recovery Test: Persistent vs Anti-persistent
    std::cout << "\nVerifying Hurst Recovery (High Hurst Proxy):" << std::endl;
    statisticalSpace.reset();
    auto highHurst = banach::test::DataGenerator::generateHurstProxy(100, 0.9f);
    for(const auto& p : highHurst) statisticalSpace.addStatisticalDataPoint({p.v, p.v, p.v}, p.t);
    statisticalSpace.performStatisticalAnalysis();

    std::cout << "\nVerifying Hurst Recovery (Low Hurst Proxy):" << std::endl;
    statisticalSpace.reset();
    auto lowHurst = banach::test::DataGenerator::generateHurstProxy(100, 0.1f);
    for(const auto& p : lowHurst) statisticalSpace.addStatisticalDataPoint({p.v, p.v, p.v}, p.t);
    statisticalSpace.performStatisticalAnalysis();

    std::cout << "exponent_stats test finished." << std::endl;
}

int main() {
    test_exponent_stats();
    return 0;
}
