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

    // Low noise parameters
    Field q(0.001, 0, 0, 0.01);
    Field r(0.01, 0, 0, 0.01);
    Field x0(0.0, 0, 0, 0.01);
    Field p0(1.0, 0, 0, 0.01);

    KalmanFieldFilter<Field> filter(q, r, p0, x0);

    double true_x = 0.0;
    double velocity = 0.5; // Constant velocity

    std::cout << "Step,TrueX,EstX,P" << std::endl;
    for(int i=0; i<50; ++i) {
        // High frequency prediction
        for(int j=0; j<5; ++j) {
            true_x += velocity * 0.02;
            filter.predict(); // Predict based on Q
        }

        // Low frequency update
        double obs_z = true_x + (random(0, 100) - 50) / 500.0;
        Field est = filter.update_linear(obs_z);

        if(i % 10 == 0) {
            std::cout << i << "," << true_x << "," << est.nominal << "," << filter.getP().nominal << std::endl;
        }

        if(i > 10) {
            assert(std::abs(est.nominal - true_x) < 0.2);
        }
    }

    std::cout << "High-Frequency Prediction test passed!" << std::endl;
    return 0;
}
