#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "../test/Arduino.h"

#include "../feature_detect/poly_feat_opt1.ino"

void test_poly_feat_opt1() {
    std::cout << "Testing AlgebraicFeatureDetector (from poly_feat_opt1.ino)..." << std::endl;

    AlgebraicFeatureDetector detector;

    // Simulate a few data points (keep within [-1, 1] to avoid GF clamping)
    for (int i = 0; i < 30; i++) {
        detector.processDataPoint(DataPoint(i * 0.05f - 0.5f, i * 0.05f - 0.5f));
    }

    auto features = detector.detectAndUpdateFeatures();
    std::cout << "Detected " << features.size() << " features." << std::endl;
    for (const auto& f : features) {
        std::cout << "Time: " << f.timestamp
                  << ", Type: " << Feature::getTypeString(f.typeIndex)
                  << ", Complexity: " << f.algebraicComplexity << std::endl;

        // Assertions for linear segment (type index 0 is linear)
        assert(f.typeIndex == 0);
    }

    std::cout << "poly_feat_opt1 test finished." << std::endl;
}

void test_quadratic_opt() {
    std::cout << "Testing quadratic segment in poly_feat_opt1..." << std::endl;
    AlgebraicFeatureDetector detector;
    // Keep quadratic within range
    for (int i = 0; i < 30; i++) {
        float t = i * 0.05f - 0.5f;
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
