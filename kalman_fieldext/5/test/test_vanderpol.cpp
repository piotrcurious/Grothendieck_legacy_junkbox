#include <iostream>
#include <cmath>
#include <cassert>
#include <iomanip>
#include "Arduino.h"
#include "mock_arduino.cpp"
#include "../GaussianDualField.h"
#include "../KalmanFieldFilter.h"

// Van der Pol oscillator simplified to 1D non-linear growth for testing predict_ekf
// dx = (mu * (1 - x^2) * x) * dt
struct VanDerPolTransition {
    double mu;
    double dt;
    GaussianDualField<double> operator()(const GaussianDualField<double>& x) const {
        return x + (x * (1.0 - x * x) * mu) * dt;
    }
};

struct LinearMeasurement {
    GaussianDualField<double> operator()(const GaussianDualField<double>& x) const {
        return x;
    }
};

int main() {
    using Field = GaussianDualField<double>;

    Field q(0.001, 0, 0, 0.01);
    Field r(0.01, 0, 0, 0.01);
    Field x0(0.5, 0, 0, 0.01);
    Field p0(0.1, 0, 0, 0.01);

    KalmanFieldFilter<Field> filter(q, r, p0, x0);
    VanDerPolTransition transition = {0.5, 0.1};

    double true_x = 0.5;

    std::cout << "Step,TrueX,EstX" << std::endl;
    for(int i=0; i<50; ++i) {
        // EKF Predict
        filter.predict_ekf(transition);

        // True state update (no noise for simplicity)
        true_x = true_x + (true_x * (1.0 - true_x * true_x) * 0.5) * 0.1;

        // EKF Update
        double obs_z = true_x + (random(0, 100) - 50) / 1000.0;
        Field est = filter.update_ekf_step(obs_z, LinearMeasurement());

        if(i % 10 == 0) {
            std::cout << i << "," << true_x << "," << est.nominal << std::endl;
        }

        if(i > 10) {
            assert(std::abs(est.nominal - true_x) < 0.1);
        }
    }

    std::cout << "Van der Pol Oscillator EKF Test passed!" << std::endl;
    return 0;
}
