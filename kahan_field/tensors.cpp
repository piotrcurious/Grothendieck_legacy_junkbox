#include "kahan_field.hpp"
#include <iostream>
#include <vector>
#include <iomanip>

void demonstrate_with_tensor_double() {
    std::cout << std::fixed << std::setprecision(16);
    std::cout << "=== Algebraic Geometry Framework (Field = Tensor<double>) ===\n\n";
    
    AlgebraicKahanSummator<Tensor<double>> kahan_tensor(Tensor<double>(2, 2, 0.0));
    
    std::vector<Tensor<double>> values = {
        Tensor<double>({{100.0, 1.0}, {0.1, 0.001}}),
        Tensor<double>({{1e-15, 2e-15}, {3e-15, 4e-15}}),
        Tensor<double>({{5e-15, 6e-15}, {7e-15, 8e-15}})
    };
    
    for (const auto& val : values) {
        kahan_tensor.add(val);
    }
    
    std::cout << "Kahan sum (s):\n" << kahan_tensor.sum() << std::endl;
    std::cout << "Kahan correction (c):\n" << kahan_tensor.correction() << std::endl;
    std::cout << "Algebraic invariant (s+c):\n" << kahan_tensor.algebraic_sum() << std::endl;
}

int main() {
    try {
        demonstrate_with_tensor_double();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
