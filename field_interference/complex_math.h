#ifndef COMPLEX_MATH_H
#define COMPLEX_MATH_H
#include <complex>
#include <vector>
#include <algorithm>
#include <random>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
typedef std::complex<double> cd;
inline std::vector<cd> dk_solve_roots(const std::vector<cd> &input_coeffs, int max_iters = 200, double tol = 1e-12) {
    std::vector<cd> coeffs = input_coeffs;
    while (coeffs.size() > 1 && std::abs(coeffs.back()) < 1e-12) coeffs.pop_back();
    int n = (int)coeffs.size() - 1;
    if (n <= 0) return {};
    std::vector<cd> roots(n);
    cd leading = coeffs[n];
    for (auto &c : coeffs) c /= leading;
    double max_a = 0.0;
    for (int i = 0; i < n; ++i) max_a = std::max(max_a, std::abs(coeffs[i]));
    double radius = 1.0 + max_a;
    static std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(0.0, 0.1);
    for (int i = 0; i < n; ++i) {
        double angle = 2.0 * M_PI * i / n + dis(gen);
        roots[i] = std::polar(radius, angle);
    }
    for (int iter = 0; iter < max_iters; ++iter) {
        double max_change = 0.0;
        for (int i = 0; i < n; ++i) {
            cd xi = roots[i];
            cd p_val = coeffs[n];
            for (int k = n - 1; k >= 0; --k) p_val = p_val * xi + coeffs[k];
            cd prod = 1.0;
            for (int j = 0; j < n; ++j) {
                if (i != j) {
                    cd diff = xi - roots[j];
                    if (std::abs(diff) < 1e-18) diff = std::polar(1e-18, 0.1);
                    prod *= diff;
                }
            }
            if (std::abs(prod) < 1e-20) prod = 1e-20;
            cd delta = p_val / prod;
            roots[i] -= delta;
            max_change = std::max(max_change, std::abs(delta));
        }
        if (max_change < tol) break;
    }
    return roots;
}
inline cd complex_eval_poly(const std::vector<cd> &coeffs, cd x) {
    cd res = 0;
    for (int i = (int)coeffs.size() - 1; i >= 0; --i) res = res * x + coeffs[i];
    return res;
}
#endif
