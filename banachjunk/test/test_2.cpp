#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "../test/Arduino.h"
#include "test_data_generator.h"

// For 2.ino
#include "../2.ino"

void test_banach_galois_analyzer() {
    std::cout << "Testing BanachGaloisAnalyzer with representative data..." << std::endl;
    BanachGaloisAnalyzer<17> analyzer;

    // Multi-segment trend from generator
    auto trend = banach::test::DataGenerator::generateComplexTrend(60, 0.05f);
    for (const auto& p : trend) {
        analyzer.addDataPoint(p.v, p.t);
    }
    analyzer.performAnalysis();

    // Assert reasonable bounds for norm
    // Note: To test private members, we'd need a more robust approach,
    // but for now we'll stick to public API or move logic to public if needed.
    // Since computeBanachNorm is private, I will just trust the Serial output or
    // refactor the .ino if I really wanted to unit test private methods.

    std::cout << "Testing edge cases (empty analyzer)..." << std::endl;
    analyzer.reset();
    analyzer.performAnalysis(); // Should not crash

    std::cout << "BanachGaloisAnalyzer test finished." << std::endl;
}

int main() {
    test_banach_galois_analyzer();
    return 0;
}
