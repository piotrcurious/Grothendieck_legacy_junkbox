#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "../test/Arduino.h"

#include "../feature_detect/poly_features.ino"

void test_poly_features() {
    std::cout << "Testing AlgebraicFeatureDetector (from poly_features.ino)..." << std::endl;

    DataPoint signalData[] = {
        // Linear segment
        {0.0, 0.1}, {0.1, 0.2}, {0.2, 0.3}, {0.3, 0.4},
        // Quadratic segment
        {0.4, 0.16}, {0.5, 0.25}, {0.6, 0.36}, {0.7, 0.49},
        // Periodic segment
        {0.8, 0.0}, {0.9, 0.866}, {1.0, 0.0}, {1.1, -0.866},
        // Complex polynomial
        {1.2, 0.5}, {1.3, -0.2}, {1.4, 0.7}, {1.5, -0.4},
        {1.6, 0.9}, {1.7, -0.6}, {1.8, 1.1}, {1.9, -0.8}
    };

    AlgebraicFeatureDetector detector;
    AlgebraicFeatureDetector::Feature features[10];

    int numFeatures = detector.detectFeatures(signalData, 20, features);

    std::cout << "Detected " << numFeatures << " features." << std::endl;
    for (int i = 0; i < numFeatures; i++) {
        std::cout << "Time: " << features[i].timestamp
                  << ", Type: " << features[i].type.c_str()
                  << ", Degree: " << features[i].polynomialDegree << std::endl;
    }
    std::cout << "AlgebraicFeatureDetector test finished." << std::endl;
}

int main() {
    test_poly_features();
    return 0;
}
