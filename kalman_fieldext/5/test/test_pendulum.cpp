#include <iostream>
#include <cmath>
#include <cassert>
#include <iomanip>
#include "Arduino.h"
#include "mock_arduino.cpp"
#include "../GaussianDualField.h"
#include "../KalmanFieldFilter.h"

struct PendulumModel {
    GaussianDualField<double> operator()(const GaussianDualField<double>& theta) const {
        return GaussianDualField<double>::sin(theta);
    }
};

int main() {
    using Field = GaussianDualField<double>;

    // Process noise Q and Measurement noise R
    Field q(0.01, 0, 0, 0.01);
    Field r(0.1, 0, 0, 0.01);
    Field x0(0.0, 0, 0, 0.01);
    Field p0(1.0, 0, 0, 0.01);

    KalmanFieldFilter<Field> filter(q, r, p0, x0);

    std::cout << "Time,True_Theta,Est_Theta,Obs_X" << std::endl;
    for(int i=0; i<100; ++i) {
        double t = i * 0.1;
        double true_theta = 0.5 * std::cos(t);
        double true_x = std::sin(true_theta);
        double obs_x = true_x + (random(0, 100) - 50) / 1000.0;

        Field est = filter.update_ekf(obs_x, PendulumModel());

        if(i % 10 == 0) {
            std::cout << t << "," << true_theta << "," << est.nominal << "," << obs_x << std::endl;
        }

        if(i > 30) {
            assert(std::abs(est.nominal - true_theta) < 0.25);
        }
    }

    std::cout << "Pendulum Tracking EKF Test passed!" << std::endl;
    return 0;
}
