#include "kahan_field.hpp"
#include <iostream>
#include <vector>
#include <iomanip>

/**
 * Demonstration of the algebraic geometry framework for Kahan summation
 */
void demonstrate_algebraic_kahan() {
    std::cout << std::fixed << std::setprecision(16);
    std::cout << "=== Algebraic Geometry Framework for Kahan Summation ===\n\n";
    
    AlgebraicKahanSummator<double> kahan;
    std::vector<double> values = {1.0, 1e-15, 1e-15, 1e-15, 1e-15, 1e-15};
    
    std::cout << "Adding values with vastly different magnitudes:\n";
    for (double val : values) {
        std::cout << "Adding: " << val << std::endl;
        kahan.add(val);
        std::cout << "  Current affine point: (" << kahan.sum() << ", " << kahan.correction() << ")" << std::endl;
        std::cout << "  Algebraic invariant: " << kahan.algebraic_sum() << std::endl;
        std::cout << std::endl;
    }
    
    double naive_sum = 0.0;
    for (double val : values) naive_sum += val;
    
    std::cout << "=== Results Comparison ===\n";
    std::cout << "Naive sum:           " << naive_sum << std::endl;
    std::cout << "Kahan sum:           " << kahan.sum() << std::endl;
    std::cout << "Algebraic invariant: " << kahan.algebraic_sum() << std::endl;
    std::cout << "Expected exact sum:  " << (1.0 + 5e-15) << std::endl;
}

int main() {
    demonstrate_algebraic_kahan();
    return 0;
}
