#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "../test/Arduino.h"
#include "test_data_generator.h"

#include "../feature_detect/poly_features.ino"

void test_poly_features() {
    std::cout << "Testing AlgebraicFeatureDetector (poly_features.ino) with representative long data..." << std::endl;

    // Construct 100-point mixed signal
    std::vector<DataPoint> signal;
    for(int i=0; i<30; ++i) signal.push_back({i*0.1f, 0.1f * i}); // Linear
    for(int i=0; i<30; ++i) {
        float t = (30+i)*0.1f;
        signal.push_back({t, 0.5f * (t-3.0f)*(t-3.0f)}); // Quadratic
    }
    auto sine = banach::test::DataGenerator::generateNoisySine(40, 1.0f, 1.0f, 0.02f, 0.1f);
    float t_start = signal.back().timestamp + 0.1f;
    for(const auto& p : sine) signal.push_back({t_start + p.t, p.v});

    AlgebraicFeatureDetector detector;
    AlgebraicFeatureDetector::Feature features[10];

    int numFeatures = detector.detectFeatures(signal.data(), signal.size(), features);

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
