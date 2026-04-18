#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "../test/Arduino.h"

// For exponent_stats.ino
#include "../exponent_stats.ino"

void test_exponent_stats() {
    std::cout << "Testing ExponentialDetector (from exponent_stats.ino)..." << std::endl;

    std::vector<float> exponentialData = {
        1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0
    };

    for (const auto& val : exponentialData) {
        expDetector.addDataPoint(val);
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
    statisticalSpace.performStatisticalAnalysis();

    // Test entropy and covariance (if public)

    std::cout << "exponent_stats test finished." << std::endl;
}

int main() {
    test_exponent_stats();
    return 0;
}
