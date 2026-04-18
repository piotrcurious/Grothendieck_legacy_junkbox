#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "../test/Arduino.h"

#include "../feature_detect/poly_feat_opt1.ino"

void test_poly_feat_opt1() {
    std::cout << "Testing AlgebraicFeatureDetector (from poly_feat_opt1.ino)..." << std::endl;

    AlgebraicFeatureDetector detector;

    // Simulate a few data points
    for (int i = 0; i < 30; i++) {
        detector.processDataPoint(DataPoint(i * 0.1f, 0.1f * i));
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

int main() {
    test_poly_feat_opt1();
    return 0;
}
