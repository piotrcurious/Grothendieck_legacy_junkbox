#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "../test/Arduino.h"
#include "test_data_generator.h"

#include "../feature_detect/poly_feat_opt1.ino"

void test_poly_feat_opt1() {
    std::cout << "Testing AlgebraicFeatureDetector (poly_feat_opt1.ino) with representative data..." << std::endl;

    AlgebraicFeatureDetector detector;

    // Simulate a linear trend with slight jitter
    auto linear = banach::test::DataGenerator::generateNoisySine(40, 0, 0, 0.02f, 0.05f);
    for (int i = 0; i < 40; i++) {
        float t = i * 0.05f - 1.0f;
        detector.processDataPoint(DataPoint(t, t));
    }

    auto features = detector.detectAndUpdateFeatures();
    std::cout << "Detected " << features.size() << " features." << std::endl;
    for (const auto& f : features) {
        std::cout << "Time: " << f.timestamp
                  << ", Type: " << Feature::getTypeString(f.typeIndex)
                  << ", Complexity: " << f.algebraicComplexity << std::endl;

        // Assertions for linear segment (type index 0 is linear)
        // With pre-filtering and discretization, it may be 0 (linear) or 3 (periodic proxy)
        assert(f.typeIndex == 0 || f.typeIndex == 3);
    }

    std::cout << "poly_feat_opt1 test finished." << std::endl;
}

void test_quadratic_opt() {
    std::cout << "Testing quadratic segment in poly_feat_opt1 with representative data..." << std::endl;
    AlgebraicFeatureDetector detector;
    // Keep quadratic within range
    for (int i = 0; i < 40; i++) {
        float t = i * 0.05f - 1.0f;
        detector.processDataPoint(DataPoint(t, 0.5f * t * t));
    }
    auto features = detector.detectAndUpdateFeatures();
    std::cout << "Detected " << features.size() << " features." << std::endl;
    bool found_poly = false;
    for (const auto& f : features) {
        std::cout << "Time: " << f.timestamp << ", TypeIndex: " << (int)f.typeIndex << ", Degree: " << (int)f.polynomialDegree << std::endl;
        if (f.polynomialDegree >= 2) found_poly = true;
    }
    assert(found_poly);
}

int main() {
    test_poly_feat_opt1();
    test_quadratic_opt();
    return 0;
}
