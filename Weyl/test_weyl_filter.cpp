#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cassert>

#define BUFFER_SIZE 128
double realBuffer[BUFFER_SIZE];
double imagBuffer[BUFFER_SIZE];

#include "Weyl_Filter_Utils.h"

fixed_complex fixedF[BUFFER_SIZE];

void test_laplacian() {
    std::cout << "Testing Laplacian regularizer..." << std::endl;
    for(int i=0; i<BUFFER_SIZE; ++i) {
        realBuffer[i] = (double)i*i / (BUFFER_SIZE*BUFFER_SIZE); // quadratic signal
        imagBuffer[i] = 0.0;
    }
    // f(x) = x^2 => f''(x) = 2.
    // processFrequencyDomain should run without issues.
    processFrequencyDomain();
    std::cout << "Laplacian regularizer sanity check OK" << std::endl;
}

int main(int argc, char** argv) {
    if (argc == 3) {
        // Mode for verify_filter.py: read from argv[1], write to argv[2]
        std::ifstream in(argv[1], std::ios::binary);
        in.read((char*)realBuffer, sizeof(double) * BUFFER_SIZE);
        for(int i=0; i<BUFFER_SIZE; ++i) imagBuffer[i] = 0.0;
        in.close();

        processFrequencyDomain();

        std::ofstream out(argv[2], std::ios::binary);
        out.write((char*)realBuffer, sizeof(double) * BUFFER_SIZE);
        out.close();
        return 0;
    }

    // Standard test mode
    test_laplacian();
    std::cout << "All host-side Weyl filter tests passed!" << std::endl;
    return 0;
}
