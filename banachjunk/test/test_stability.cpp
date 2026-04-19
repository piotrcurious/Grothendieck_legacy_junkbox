#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "Arduino.h"
#include "../math_utils.h"

void test_ridge_stability() {
    std::cout << "Testing Ridge Regression Stability..." << std::endl;

    // Case 1: Perfect linear signal
    std::vector<double> x = {0, 1, 2, 3, 4, 5};
    std::vector<double> y = {1, 2, 3, 4, 5, 6}; // y = x + 1
    std::vector<double> dt = {1, 1, 1, 1, 1, 1};

    auto res = banach::LinearSolvers::solveRidge<2>(x, y, dt, 1e-6);
    assert(res.success);
    std::cout << "Linear Fit: " << res.coefficients[1] << "x + " << res.coefficients[0] << std::endl;
    assert(std::abs(res.coefficients[1] - 1.0) < 1e-3);
    assert(std::abs(res.coefficients[0] - 1.0) < 1e-3);

    // Case 2: Ill-conditioned system (redundant data)
    std::vector<double> x_ill = {1, 1, 1, 1};
    std::vector<double> y_ill = {5, 5, 5, 5};
    auto res_ill = banach::LinearSolvers::solveRidge<2>(x_ill, y_ill, dt, 1e-6);
    std::cout << "Ill-conditioned Condition Proxy: " << res_ill.conditionProxy << std::endl;
    // With Ridge, it should still "solve" (give reasonable constant part)
    assert(res_ill.success);
    assert(res_ill.conditionProxy > 1e6);
}

void test_moments_consistency() {
    std::cout << "Testing Moment Consistency..." << std::endl;
    std::vector<double> values = {10, 10, 10, 10};
    std::vector<double> ts = {0, 1, 2, 3};
    auto m = banach::Statistics::calculateMoments(values, ts);
    assert(std::abs(m.mean - 10.0) < 1e-6);
    assert(std::abs(m.variance) < 1e-6);
}

int main() {
    test_ridge_stability();
    test_moments_consistency();
    std::cout << "Stability tests passed!" << std::endl;
    return 0;
}
