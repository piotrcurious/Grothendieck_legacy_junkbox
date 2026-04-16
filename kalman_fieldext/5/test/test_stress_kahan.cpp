#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include "../GaussianDualField.h"

int main() {
    using Field = GaussianDualField<double>;
    double initial = 1e12;
    double small = 1e-5;
    long iterations = 100000000; // 100 million

    Field kahan_sum(initial, 0, 0, 0.01);
    double naive_sum = initial;

    std::cout << "Starting stress test with " << iterations << " iterations..." << std::endl;
    for (long i = 0; i < iterations; ++i) {
        kahan_sum = kahan_sum + Field(small, 0, 0, 0.01);
        naive_sum += small;
    }

    double expected = initial + (double)iterations * small;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Expected:  " << expected << std::endl;
    std::cout << "Kahan sum: " << kahan_sum.nominal << " (Error: " << std::abs(kahan_sum.nominal - expected) << ")" << std::endl;
    std::cout << "Naive sum: " << naive_sum << " (Error: " << std::abs(naive_sum - expected) << ")" << std::endl;

    if (std::abs(kahan_sum.nominal - expected) < std::abs(naive_sum - expected)) {
        std::cout << "Kahan summation is significantly more accurate!" << std::endl;
    } else if (std::abs(kahan_sum.nominal - expected) == std::abs(naive_sum - expected)) {
        std::cout << "Both methods produced same result (check if precision was enough to show difference)." << std::endl;
    } else {
        std::cout << "WARNING: Naive sum was more accurate? Check logic." << std::endl;
    }

    return 0;
}
