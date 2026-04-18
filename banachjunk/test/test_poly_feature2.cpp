#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "../test/Arduino.h"

#include "../feature_detect/poly_feature2.ino"

void test_poly_feature2() {
    std::cout << "Testing AlgebraicFeatureDetector (from poly_feature2.ino)..." << std::endl;

    AlgebraicFeatureDetector detector;

    // Simulate some data
    for (int i = 0; i < 40; i++) {
        detector.processDataPoint(DataPoint(i * 0.1f, 0.1f * i));
    }

    auto features = detector.detectAndUpdateFeatures();
    std::cout << "Detected " << features.size() << " features." << std::endl;
    for (const auto& f : features) {
        std::cout << "Time: " << f.timestamp
                  << ", Type: " << f.type
                  << ", Degree: " << f.polynomialDegree
                  << ", Confidence: " << f.confidence << std::endl;
    }

    std::cout << "poly_feature2 test finished." << std::endl;
}

int main() {
    test_poly_feature2();
    return 0;
}
