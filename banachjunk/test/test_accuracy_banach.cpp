#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <cassert>
#include "Arduino.h"
#include "test_data_generator.h"

#include "../feature_detect/feature_detect.ino"

void test_accuracy_banach() {
    std::cout << "Evaluating BanachFeatureDetector Accuracy..." << std::endl;
    auto ls = banach::test::DataGenerator::generateLabeledSignal();

    BanachFeatureDetector bfd;
    BanachFeatureDetector::Feature features[10];

    std::vector<DataPoint> signal;
    for(const auto& p : ls.signal) signal.push_back({p.t, p.v});

    int count = bfd.detectFeatures(signal.data(), signal.size(), features);

    int matches = 0;
    auto isMatch = [](const std::string& detected, const std::string& truth) {
        if (detected == truth) return true;
        if (detected == "transition" && truth == "linear") return true;
        if (detected == "oscillation" && truth == "periodic") return true;
        if (detected == "trend" && truth == "linear") return true;
        if (detected == "step" && truth == "linear") return true;
        return false;
    };

    for (int i = 0; i < count; ++i) {
        std::string d_type = features[i].type.c_str();
        float d_t = features[i].timestamp;
        std::cout << "Detected: " << d_type << " at t=" << d_t << std::endl;

        for (const auto& t : ls.truth) {
            if (std::abs(d_t - t.t) < 2.5f && isMatch(d_type, t.type)) {
                matches++;
                break;
            }
        }
    }
    std::cout << "Banach Accuracy: " << matches << "/" << ls.truth.size() << " matched." << std::endl;
    assert(matches >= 2);
}

int main() {
    test_accuracy_banach();
    return 0;
}
