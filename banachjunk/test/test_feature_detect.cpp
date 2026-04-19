#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "../test/Arduino.h"
#include "test_data_generator.h"

#include "../feature_detect/feature_detect.ino"

void test_feature_detect() {
    std::cout << "Testing BanachFeatureDetector with complex physical representative data..." << std::endl;

    // Construct complex signal: Constant -> Chirp -> Spike -> StepRelaxation
    std::vector<DataPoint> signal;
    for(int i=0; i<50; ++i) signal.push_back({i*0.1f, 0.1f}); // Constant

    auto chirp = banach::test::DataGenerator::generateChirp(100, 0.1f, 2.0f, 10.0f, 1.0f);
    float t_offset = signal.back().timestamp + 0.1f;
    for(const auto& p : chirp) signal.push_back({t_offset + p.t, p.v});

    signal.push_back({signal.back().timestamp + 0.1f, 10.0f}); // Huge Spike

    auto step = banach::test::DataGenerator::generateStepRelaxation(100, 0, 5.0f, 2.0f);
    t_offset = signal.back().timestamp + 0.1f;
    for(const auto& p : step) signal.push_back({t_offset + p.t, p.v});

    BanachFeatureDetector detector;
    BanachFeatureDetector::Feature features[10];

    int numFeatures = detector.detectFeatures(signal.data(), signal.size(), features);

    std::cout << "Detected " << numFeatures << " features." << std::endl;
    for (int i = 0; i < numFeatures; i++) {
        std::cout << "Time: " << features[i].timestamp
                  << ", Type: " << features[i].type.c_str()
                  << ", Strength: " << features[i].strength << std::endl;
    }
    std::cout << "BanachFeatureDetector test finished." << std::endl;
}

int main() {
    test_feature_detect();
    return 0;
}
