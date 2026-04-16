#include <iostream>
#include <cmath>
#include <cassert>
#include <iomanip>
#include "Arduino.h"
#include "mock_arduino.cpp"
#include "../GaussianDualField.h"
#include "../KalmanFieldFilter.h"

int main() {
    using Field = GaussianDualField<double>;

    // Process noise Q and Measurement noise R
    Field q(0.01, 0, 0, 0.01);
    Field r(0.1, 0, 0, 0.01);

    // Initial state: x=1.0, P=1.0
    Field x(1.0, 0, 0, 0.01);
    Field p(1.0, 0, 0, 0.01);

    double true_x = 1.0;

    std::cout << "Step,TrueX,EstX,ObsZ,Gain" << std::endl;
    for(int i=0; i<50; ++i) {
        // 1. Predict
        p = p + q;

        // 2. Nonlinear Measurement z = sin(x)
        // To use Dual Numbers for Jacobian, we set x.delta = 1.0 before evaluating H(x)
        x.delta = 1.0;
        Field h_x = Field::sin(x);
        double z_pred = h_x.nominal;
        double H = h_x.delta; // dh/dx = cos(x)

        // 3. Update
        double true_z = std::sin(true_x);
        double obs_z = true_z + (random(0, 100) - 50) / 500.0;

        double K = p.nominal * H / (H * p.nominal * H + r.nominal);
        x.nominal = x.nominal + K * (obs_z - z_pred);
        p.nominal = (1.0 - K * H) * p.nominal; // Simple update for nominal P

        if(i % 10 == 0) {
            std::cout << i << "," << true_x << "," << x.nominal << "," << obs_z << "," << K << std::endl;
        }

        // Random walk for true state
        true_x += (random(0, 100) - 50) / 500.0;
    }

    std::cout << "EKF Example (Manual Jacobian via Dual Numbers) finished!" << std::endl;
    return 0;
}
