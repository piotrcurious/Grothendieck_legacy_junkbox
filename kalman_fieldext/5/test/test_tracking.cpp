#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include "Arduino.h"
#include "mock_arduino.cpp"
#include "../GaussianDualField.h"
#include "../KalmanFieldFilter.h"

int main() {
    using Field = GaussianDualField<double>;

    // Higher Q to track dynamic signal
    KalmanFieldFilter<Field> filter(
        Field(0.1, 0, 0, 0.01),
        Field(0.01, 0, 1.0, 0.01), // delta tracks sensitivity to R
        Field(1.0, 0, 0, 0.01),
        Field(0.0, 0, 0, 0.01)
    );

    std::cout << "Time,True,Est,Sensitivity_to_R" << std::endl;
    for (int i = 0; i < 100; ++i) {
        double t = i * 0.1;
        double true_val = 5.0 + 2.0 * t; // Ramp signal
        double z = true_val + (random(0, 100) - 50) / 100.0;

        Field x = filter.update_field(z);
        if (i % 10 == 0) {
            std::cout << t << "," << true_val << "," << x.nominal << "," << x.delta << std::endl;
        }

        // Assert convergence to ramp
        if (i > 50) {
            assert(std::abs(x.nominal - true_val) < 1.0);
        }
    }

    std::cout << "Dynamic tracking test passed!" << std::endl;
    return 0;
}
