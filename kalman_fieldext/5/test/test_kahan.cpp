#include <iostream>
#include <iomanip>
#include <vector>
#include "../GaussianDualField.h"

int main() {
    using Field = GaussianDualField<float>;
    float val = 1.0f;
    float small = 1e-7f;

    Field sum(val, 0, 0, 0.01f);
    float simple_sum = val;

    for(int i=0; i<1000000; ++i) {
        sum = sum + Field(small, 0, 0, 0.01f);
        simple_sum += small;
    }

    std::cout << std::fixed << std::setprecision(10);
    std::cout << "Kahan sum: " << sum.nominal << std::endl;
    std::cout << "Simple sum: " << simple_sum << std::endl;
    std::cout << "Expected: " << (val + 1000000 * small) << std::endl;

    return 0;
}
