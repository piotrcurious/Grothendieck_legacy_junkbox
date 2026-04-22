#ifndef COMPLEX_MATH_H
#define COMPLEX_MATH_H

#include <complex>
#include <vector>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef std::complex<double> cd;

/**
 * Durand-Kerner method for finding all roots of a polynomial simultaneously.
 * Coeffs are ordered: a_0 + a_1*x + ... + a_n*x^n
 */
inline std::vector<cd> dk_solve_roots(const std::vector<cd> &input_coeffs, int max_iters = 200, double tol = 1e-12) {
    std::vector<cd> coeffs = input_coeffs;
    // Strip trailing zeros to find actual degree
    while (coeffs.size() > 1 && std::abs(coeffs.back()) < 1e-12) {
        coeffs.pop_back();
    }

    int n = (int)coeffs.size() - 1;
    if (n <= 0) return {};
    std::vector<cd> roots(n);

    // Normalize coefficients so leading term is 1
    cd leading = coeffs[n];
    for (auto &c : coeffs) c /= leading;

    // Cauchy bound for initial radius
    double max_a = 0.0;
    for (int i = 0; i < n; ++i) max_a = std::max(max_a, std::abs(coeffs[i]));
    double radius = 1.0 + max_a;

    // Initialize roots on a circle with slight offset to avoid symmetry locks
    for (int i = 0; i < n; ++i) {
        roots[i] = std::polar(radius, 2.0 * M_PI * i / n + 0.1);
    }

    for (int iter = 0; iter < max_iters; ++iter) {
        double max_change = 0.0;
        for (int i = 0; i < n; ++i) {
            cd xi = roots[i];
            // Horner's method for P(xi)
            cd p_val = coeffs[n];
            for (int k = n - 1; k >= 0; --k) p_val = p_val * xi + coeffs[k];

            cd prod = 1.0;
            for (int j = 0; j < n; ++j) {
                if (i != j) prod *= (xi - roots[j]);
            }

            if (std::abs(prod) < 1e-18) prod = 1e-18;
            cd delta = p_val / prod;
            roots[i] -= delta;
            max_change = std::max(max_change, std::abs(delta));
        }
        if (max_change < tol) break;
    }
    return roots;
}

/**
 * Evaluates a polynomial with complex coefficients at a given point.
 */
inline cd complex_eval_poly(const std::vector<cd> &coeffs, cd x) {
    cd res = 0;
    for (int i = (int)coeffs.size() - 1; i >= 0; --i) {
        res = res * x + coeffs[i];
    }
    return res;
}

#endif
