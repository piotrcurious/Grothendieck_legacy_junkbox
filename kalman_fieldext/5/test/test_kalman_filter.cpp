#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include "Arduino.h"
#include "mock_arduino.cpp"
#include "../GaussianDualField.h"
#include "../KalmanFieldFilter.h"

void test_filter_convergence() {
    using Field = GaussianDualField<double>;
    KalmanFieldFilter<Field> filter(
        Field(0.001, 0, 0, 0.01),
        Field(0.01, 0, 0, 0.01),
        Field(1.0, 0, 0, 0.01),
        Field(0.0, 0, 0, 0.01)
    );
    double true_val = 10.0;
    for (int i = 0; i < 100; ++i) {
        double z = true_val + (random(0, 100) - 50) / 500.0;
        filter.update(z);
    }
    double final_est = filter.update(true_val);
    std::cout << "Final estimate: " << final_est << std::endl;
    assert(std::abs(final_est - true_val) < 0.1);
}

void test_sensitivity() {
    using Field = GaussianDualField<double>;
    double r0 = 0.01;
    KalmanFieldFilter<Field> filter(
        Field(0.001, 0, 0, 0.01),
        Field(r0, 0, 1.0, 0.01),
        Field(1.0, 0, 0, 0.01),
        Field(0.0, 0, 0, 0.01)
    );
    double true_val = 10.0;
    double last_delta = 0;
    for (int i = 0; i < 50; ++i) {
        Field x_field = filter.update_field(true_val);
        last_delta = x_field.delta;
    }
    double eps = 1e-6;
    KalmanFieldFilter<Field> filter1(Field(0.001, 0, 0, 0.01), Field(r0, 0, 0, 0.01), Field(1.0, 0, 0, 0.01), Field(0.0, 0, 0, 0.01));
    KalmanFieldFilter<Field> filter2(Field(0.001, 0, 0, 0.01), Field(r0 + eps, 0, 0, 0.01), Field(1.0, 0, 0, 0.01), Field(0.0, 0, 0, 0.01));
    for(int i=0; i<50; ++i) { filter1.update(true_val); filter2.update(true_val); }
    double numerical = (filter2.update(true_val) - filter1.update(true_val)) / eps;
    std::cout << "dx/dR: " << last_delta << " (dual), " << numerical << " (numerical)" << std::endl;
    assert(std::abs(last_delta - numerical) < 1e-3);
}

void test_noise_propagation() {
    using Field = GaussianDualField<double>;
    double sigma2 = 0.01;
    KalmanFieldFilter<Field> filter(
        Field(0.001, 0.1, 0, sigma2),
        Field(0.01, 0, 0, sigma2),
        Field(1.0, 0, 0, sigma2),
        Field(0.0, 0, 0, sigma2)
    );
    double true_val = 10.0;
    for (int i = 0; i < 50; ++i) {
        Field x_field = filter.update_field(true_val);
        if (i % 20 == 0) std::cout << "Step " << i << ": x=" << x_field.nominal << ", noise=" << x_field.noise << std::endl;
    }
}

int main() {
    test_filter_convergence();
    test_sensitivity();
    test_noise_propagation();
    std::cout << "Kalman Field Filter tests passed!" << std::endl;
    return 0;
}
