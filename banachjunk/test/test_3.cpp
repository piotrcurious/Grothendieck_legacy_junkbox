#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "../test/Arduino.h"
#include "test_data_generator.h"

// For 3_space.ino
#include "../3_space.ino"

void test_banach_space() {
    std::cout << "Testing BanachSpace with representative jittered data..." << std::endl;
    BanachSpace<float, 3> numericalSpace;

    // Generate complex multi-dimensional signal with jitter and varying correlation
    auto [s1, s2] = banach::test::DataGenerator::generateCorrelatedSignals(200, 0.05f);
    auto s3 = banach::test::DataGenerator::generateChirp(200, 0.1f, 2.0f, 20.0f, 1.0f);

    for (int i = 0; i < 200; ++i) {
        numericalSpace.addDataPoint({s1[i].v, s2[i].v, s3[i].v}, s1[i].t);
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
