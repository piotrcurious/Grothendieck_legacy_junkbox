#include <iostream>
#include <cmath>
#include <cassert>
#include <iomanip>
#include "Arduino.h"
#include "mock_arduino.cpp"
#include "../GaussianDualField.h"
#include "../KalmanFieldFilter.h"

struct MeasurementModel {
    GaussianDualField<double> operator()(const GaussianDualField<double>& x) const {
        return GaussianDualField<double>::sin(x);
    }
};

int main() {
    using Field = GaussianDualField<double>;

    Field q(0.01, 0, 0, 0.01);
    Field r(0.1, 0, 0, 0.01);
    Field x0(1.0, 0, 0, 0.01);
    Field p0(1.0, 0, 0, 0.01);

    KalmanFieldFilter<Field> filter(q, r, p0, x0);
    double true_x = 1.0;

    std::cout << "Step,TrueX,EstX,ObsZ" << std::endl;
    for(int i=0; i<50; ++i) {
        double true_z = std::sin(true_x);
        double obs_z = true_z + (random(0, 100) - 50) / 500.0;

        Field est = filter.update_ekf(obs_z, MeasurementModel());

        if(i % 10 == 0) {
            std::cout << i << "," << true_x << "," << est.nominal << "," << obs_z << std::endl;
        }

        true_x += (random(0, 100) - 50) / 500.0;
    }

    std::cout << "Automated EKF Test finished!" << std::endl;
    return 0;
}
