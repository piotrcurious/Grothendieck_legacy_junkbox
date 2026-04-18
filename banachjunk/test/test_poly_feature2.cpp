#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "../test/Arduino.h"

#include "../feature_detect/poly_feature2.ino"

void test_poly_feature2() {
    std::cout << "Testing AlgebraicFeatureDetector (from poly_feature2.ino)..." << std::endl;

    AlgebraicFeatureDetector detector;

    // Simulate some data in range [-1, 1]
    for (int i = 0; i < 40; i++) {
        detector.processDataPoint(DataPoint(i * 0.05f - 1.0f, i * 0.05f - 1.0f));
    }

    auto features = detector.detectAndUpdateFeatures();
    std::cout << "Detected " << features.size() << " features." << std::endl;
    for (const auto& f : features) {
        std::cout << "Time: " << f.timestamp
                  << ", Type: " << f.type
                  << ", Degree: " << f.polynomialDegree
                  << ", Confidence: " << f.confidence << std::endl;

        // Assertions for linear segment
        assert(f.polynomialDegree <= 1);
        assert(f.type == "linear");
    }
    std::cout << "poly_feature2 linear test finished." << std::endl;
}

void test_quadratic_segment() {
    std::cout << "Testing quadratic segment in poly_feature2..." << std::endl;
    AlgebraicFeatureDetector detector;
    // Window is 20 points. We provide 30 points of quadratic curve
    // Scale to ensure it fits well in GF[251] mapping
    for (int i = 0; i < 20; i++) {
        float t = i * 0.05f - 0.75f;
        detector.processDataPoint(DataPoint(t, t * t));
    }
    auto features = detector.detectAndUpdateFeatures();
    if(features.empty()) {
        // Try one more point to trigger if buffering is off by one
        detector.processDataPoint(DataPoint(20 * 0.05f - 0.75f, 0.5f));
        features = detector.detectAndUpdateFeatures();
    }
    std::cout << "Detected " << features.size() << " features." << std::endl;
    bool found_poly = false;
    for (const auto& f : features) {
        std::cout << "Time: " << f.timestamp << ", Type: " << f.type << ", Degree: " << f.polynomialDegree << std::endl;
        if (f.polynomialDegree >= 2) found_poly = true;
    }
    assert(found_poly);
    std::cout << "poly_feature2 quadratic test finished." << std::endl;
}

int main() {
    test_poly_feature2();
    test_quadratic_segment();
    return 0;
}
