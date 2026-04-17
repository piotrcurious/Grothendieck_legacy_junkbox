#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "../test/Arduino.h"

// For 2.ino
#include "../2.ino"

void test_banach_galois_analyzer() {
    std::cout << "Testing BanachGaloisAnalyzer (from 2.ino)..." << std::endl;
    BanachGaloisAnalyzer<17> analyzer;
    float simulatedData[] = {
        1.2, 2.4, 4.8, 9.6, 19.2,
        3.5, 7.0, 14.0, 28.0,
        2.1, 4.2, 8.4, 16.8
    };
    for (float data : simulatedData) {
        analyzer.addDataPoint(data);
    }
    analyzer.performAnalysis();
    std::cout << "BanachGaloisAnalyzer test finished." << std::endl;
}

int main() {
    test_banach_galois_analyzer();
    return 0;
}
