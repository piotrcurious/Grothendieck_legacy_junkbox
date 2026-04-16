#include <iostream>
#include <cassert>
#include <cmath>
#include <iomanip>
#include "../GaussianDualField.h"

void test_exp_log() {
    using Field = GaussianDualField<double>;
    Field x(1.5, 0.2, 0.3, 0.01);

    Field ex = Field::exp(x);
    Field lex = Field::log(ex);

    std::cout << std::fixed << std::setprecision(10);
    std::cout << "Original: nominal=" << x.nominal << ", noise=" << x.noise << ", delta=" << x.delta << std::endl;
    std::cout << "Log(Exp): nominal=" << lex.nominal << ", noise=" << lex.noise << ", delta=" << lex.delta << std::endl;

    assert(std::abs(lex.nominal - x.nominal) < 1e-9);
    assert(std::abs(lex.noise - x.noise) < 1e-9);
    // Delta in log(exp(x)) is c, so we allow some tolerance for higher-order effects if any,
    // but here it should be quite close.
    assert(std::abs(lex.delta - x.delta) < 1e-3);

    std::cout << "Exp/Log tests passed!" << std::endl;
}

void test_sqrt() {
    using Field = GaussianDualField<double>;
    Field x(2.0, 0.5, 0.4, 0.01);

    Field sx = Field::sqrt(x);
    Field sx2 = sx * sx;

    std::cout << "Original: nominal=" << x.nominal << ", noise=" << x.noise << ", delta=" << x.delta << std::endl;
    std::cout << "Sqrt^2:   nominal=" << sx2.nominal << ", noise=" << sx2.noise << ", delta=" << sx2.delta << std::endl;

    assert(std::abs(sx2.nominal - x.nominal) < 1e-9);
    assert(std::abs(sx2.noise - x.noise) < 1e-9);
    assert(std::abs(sx2.delta - x.delta) < 1e-3);

    std::cout << "Sqrt tests passed!" << std::endl;
}

void test_derivatives() {
    using Field = GaussianDualField<double>;
    double val = 2.0;
    double eps = 1e-7;

    // Test exp derivative via delta
    Field x_exp(val, 0, 1.0, 0.01);
    Field res_exp = Field::exp(x_exp);
    double dual_der_exp = res_exp.delta;
    double num_der_exp = (std::exp(val + eps) - std::exp(val)) / eps;

    std::cout << "Exp der: dual=" << dual_der_exp << ", num=" << num_der_exp << std::endl;
    assert(std::abs(dual_der_exp - num_der_exp) < 1e-5);

    // Test log derivative
    Field x_log(val, 0, 1.0, 0.01);
    Field res_log = Field::log(x_log);
    double dual_der_log = res_log.delta;
    double num_der_log = (std::log(val + eps) - std::log(val)) / eps;

    std::cout << "Log der: dual=" << dual_der_log << ", num=" << num_der_log << std::endl;
    assert(std::abs(dual_der_log - num_der_log) < 1e-5);

    std::cout << "Derivative tests passed!" << std::endl;
}

int main() {
    test_exp_log();
    test_sqrt();
    test_derivatives();
    return 0;
}
