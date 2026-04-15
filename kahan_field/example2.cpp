#include "kahan_field.hpp"
#include <iostream>
#include <vector>
#include <iomanip>

void demonstrate_with_double() {
    std::cout << std::fixed << std::setprecision(16);
    std::cout << "=== Algebraic Geometry Framework (Field = double) ===\n\n";
    
    AlgebraicKahanSummator<double> kahan_double;
    std::vector<double> values = {1.0, 1e-15, 1e-15, 1e-15, 1e-15, 1e-15};
    
    for (double val : values) {
        kahan_double.add(val);
    }
    
    std::cout << "Kahan sum (s):         " << kahan_double.sum() << std::endl;
    std::cout << "Kahan correction (c):  " << kahan_double.correction() << std::endl;
    std::cout << "Algebraic invariant:   " << kahan_double.algebraic_sum() << std::endl;
}

void demonstrate_with_gf_p() {
    std::cout << "\n=== Algebraic Geometry Framework (Field = GF_p<5>) ===\n\n";
    
    AlgebraicKahanSummator<GF_p<5>> kahan_gf;
    std::vector<long long> raw_values = {1, 2, 3, 4, 1, 2};
    
    for (long long raw_val : raw_values) {
        kahan_gf.add(GF_p<5>(raw_val));
    }
    
    std::cout << "Kahan sum (s):         " << kahan_gf.sum() << std::endl;
    std::cout << "Kahan correction (c):  " << kahan_gf.correction() << " (always 0 in exact fields)\n";
    std::cout << "Algebraic invariant:   " << kahan_gf.algebraic_sum() << std::endl;
}

int main() {
    demonstrate_with_double();
    demonstrate_with_gf_p();
    return 0;
}
