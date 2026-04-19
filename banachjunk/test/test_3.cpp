#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "../test/Arduino.h"

// For 3_space.ino
#include "../3_space.ino"

void test_banach_space() {
    std::cout << "Testing BanachSpace (from 3_space.ino)..." << std::endl;
    BanachSpace<float, 3> numericalSpace;
    std::vector<std::vector<float>> simulatedData = {
        {1.2, 2.4, 4.8, 9.6, 19.2},   // Dimension 1
        {3.5, 7.0, 14.0, 28.0, 56.0},  // Dimension 2
        {2.1, 4.2, 8.4, 16.8, 33.6}    // Dimension 3
    };

    for (size_t i = 0; i < simulatedData[0].size(); ++i) {
        std::vector<float> point = {
            simulatedData[0][i],
            simulatedData[1][i],
            simulatedData[2][i]
        };
        numericalSpace.addDataPoint(point);
    }
    numericalSpace.performSpaceAnalysis();

    // Rigorous assertions for 3rd iteration metrics
    float l2 = numericalSpace.computeLpNorm(2);
    assert(l2 > 0);

    // Verify spectral flatness range [0, 1]
    // Flatness is only available inside performSpaceAnalysis output,
    // but we can check internal logic via a public wrapper or re-run
    std::cout << "Verifying Spectral Flatness..." << std::endl;
    // (Spectral flatness was added as a public method computeSpectralFlatness in 3_space.ino)
    auto flatness = numericalSpace.computeSpectralFlatness();
    for(float f : flatness) {
        assert(f >= 0.0f && f <= 1.0f);
    }

    std::cout << "Verifying Instantaneous Coherence..." << std::endl;
    auto instCovar = numericalSpace.computeInstantaneousCoherence(3);
    assert(instCovar.size() == 3);
    for(size_t i=0; i<3; ++i) assert(instCovar[i][i] >= 0.99f); // Self-coherence

    std::cout << "Testing edge cases (empty space)..." << std::endl;
    numericalSpace.reset();
    numericalSpace.performSpaceAnalysis(); // Should not crash

    std::cout << "BanachSpace test finished." << std::endl;
}

int main() {
    test_banach_space();
    return 0;
}
