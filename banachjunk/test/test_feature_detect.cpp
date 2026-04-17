#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "../test/Arduino.h"

#include "../feature_detect/feature_detect.ino"

void test_feature_detect() {
    std::cout << "Testing BanachFeatureDetector (from feature_detect.ino)..." << std::endl;

    DataPoint signalData[] = {
        {0.0, 0.1}, {0.1, 0.15}, {0.2, 0.12}, {0.3, 0.8},  // Step
        {0.4, 0.85}, {0.5, 0.82}, {0.6, 0.81}, {0.7, 0.79},
        {0.8, 0.2}, {0.9, 0.18}, {1.0, 1.5}, {1.1, 0.2},   // Spike
        {1.2, 0.22}, {1.3, 0.19}, {1.4, 0.21}, {1.5, 0.18},
        {1.6, 0.3}, {1.7, -0.3}, {1.8, 0.3}, {1.9, -0.3},  // Oscillation
        {2.0, 0.3}, {2.1, -0.3}, {2.2, 0.3}, {2.3, -0.3}
    };

    BanachFeatureDetector detector;
    BanachFeatureDetector::Feature features[10];

    int numFeatures = detector.detectFeatures(signalData, 24, features);

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
