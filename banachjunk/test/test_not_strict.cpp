#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "../test/Arduino.h"

// For not_strict.ino
#include "../not_strict.ino"

void test_not_strict() {
    std::cout << "Testing BanachSpaceAnalyzer (from not_strict.ino)..." << std::endl;
    // Note: BanachSpaceAnalyzer name is reused from other files, but here it's different class.
    // In this context it refers to the one in not_strict.ino
    float simulatedData[] = {1.0, 2.2, 4.1, 7.5, 12.3, 18.9, 27.4};
    for (float data : simulatedData) {
        banachAnalyzer.addData(data);
    }
    banachAnalyzer.analyzeBanachSpace();

    // Test polynomial evaluation (if public)

    std::cout << "BanachSpaceAnalyzer test finished." << std::endl;
}

int main() {
    test_not_strict();
    return 0;
}
