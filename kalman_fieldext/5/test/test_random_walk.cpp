#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <random>
#include "Arduino.h"
#include "mock_arduino.cpp"
#include "../GaussianDualField.h"
#include "../KalmanFieldFilter.h"

int main() {
    using Field = GaussianDualField<double>;

    Field q(0.1, 0, 0, 0.01);
    Field r(0.5, 0, 0, 0.01);
    Field p0(1.0, 0, 0, 0.01);
    Field x0(0.0, 0, 0, 0.01);

    KalmanFieldFilter<Field> filter(q, r, p0, x0);

    std::default_random_engine generator(42);
    std::normal_distribution<double> dist_q(0.0, std::sqrt(q.nominal));
    std::normal_distribution<double> dist_r(0.0, std::sqrt(r.nominal));

    double true_x = 0.0;
    std::cout << "Step,True,Est,Error" << std::endl;
    for (int i = 0; i < 100; ++i) {
        true_x += dist_q(generator);
        double z = true_x + dist_r(generator);

        double est = filter.update(z);
        if (i % 10 == 0) {
            std::cout << i << "," << true_x << "," << est << "," << (est - true_x) << std::endl;
        }
    }

    std::cout << "Random walk test passed!" << std::endl;
    return 0;
}
