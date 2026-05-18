#include "Arduino.h"
#include "../FieldExtension.h"
#include <iostream>
#include <chrono>

int main() {
    std::cout << "Running Benchmark Test..." << std::endl;

    std::cout << "\n--- Multiplication Benchmark (10,000 ops) ---" << std::endl;
    FieldElement<32> a;
    for(int i=0; i<32; i++) a.setCoefficient(i, (float)i/100.0f);
    FieldElement<32> b = a;

    auto start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<10000; i++) {
        a = a * b;
        if (a.norm() > 10.0f) a *= 0.1f;
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Time: " << diff.count() * 1e6 << " us" << std::endl;

    std::cout << "\n--- Transcendental Benchmark (1,000 ops) ---" << std::endl;
    FieldElement16 x(0.5f);
    FieldElement16 res;
    start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<1000; i++) {
        res = sin(x);
        x = res * 0.5f + 0.1f;
    }
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    std::cout << "Sin Time: " << diff.count() * 1e6 << " us" << std::endl;

    std::cout << "Benchmark Test Completed Successfully!" << std::endl;
    return 0;
}
