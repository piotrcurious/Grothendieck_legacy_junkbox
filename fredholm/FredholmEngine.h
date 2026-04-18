#ifndef FREDHOLM_ENGINE_H
#define FREDHOLM_ENGINE_H

#include <vector>
#include <cmath>
#include <functional>
#include <complex>
#include <algorithm>
#include <iostream>

namespace Fredholm {

/**
 * @brief Quadrature points and weights for numerical integration
 */
struct Quadrature {
    std::vector<double> points;
    std::vector<double> weights;

    static Quadrature GaussLegendre8() {
        return {
            {-0.9602898564975363, -0.7966664774136267, -0.5255324099163290, -0.1834346424956498,
              0.1834346424956498,  0.5255324099163290,  0.7966664774136267,  0.9602898564975363},
            { 0.1012285362903763,  0.2223810344533745,  0.3137066458778873,  0.3626837833783620,
              0.3626837833783620,  0.3137066458778873,  0.2223810344533745,  0.1012285362903763}
        };
    }

    static Quadrature GaussLegendreN(int n, double a, double b) {
        // Simple scaling of 8-point quadrature
        Quadrature q = GaussLegendre8();
        for (size_t i = 0; i < q.points.size(); ++i) {
            double p = q.points[i];
            q.points[i] = 0.5 * (b - a) * p + 0.5 * (a + b);
            q.weights[i] *= 0.5 * (b - a);
        }
        return q;
    }
};

/**
 * @brief Linear system solver using Gaussian elimination with partial pivoting
 */
inline bool solveLinearSystem(std::vector<std::vector<double>>& A, std::vector<double>& b, std::vector<double>& x) {
    int n = b.size();
    for (int i = 0; i < n; i++) {
        double maxEl = std::abs(A[i][i]);
        int maxRow = i;
        for (int k = i + 1; k < n; k++) {
            if (std::abs(A[k][i]) > maxEl) {
                maxEl = std::abs(A[k][i]);
                maxRow = k;
            }
        }

        std::swap(A[maxRow], A[i]);
        std::swap(b[maxRow], b[i]);

        for (int k = i + 1; k < n; k++) {
            if (std::abs(A[i][i]) < 1e-15) return false;
            double c = -A[k][i] / A[i][i];
            for (int j = i; j < n; j++) {
                if (i == j) A[k][j] = 0;
                else A[k][j] += c * A[i][j];
            }
            b[k] += c * b[i];
        }
    }

    x.resize(n);
    for (int i = n - 1; i >= 0; i--) {
        if (std::abs(A[i][i]) < 1e-15) return false;
        x[i] = b[i] / A[i][i];
        for (int k = i - 1; k >= 0; k--) {
            b[k] -= A[k][i] * x[i];
        }
    }
    return true;
}

/**
 * @brief General solver for Fredholm integral equations of the second kind:
 * phi(x) = f(x) + lambda * integral_a^b K(x, y) phi(y) dy
 */
class Solver {
public:
    using KernelFunc = std::function<double(double, double)>;
    using SourceFunc = std::function<double(double)>;

    static std::vector<double> solve(double a, double b, double lambda,
                                     KernelFunc K, SourceFunc f,
                                     int n = 8) {
        Quadrature q = Quadrature::GaussLegendreN(n, a, b);
        int N = q.points.size();

        std::vector<std::vector<double>> A(N, std::vector<double>(N));
        std::vector<double> B(N);

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                double delta = (i == j) ? 1.0 : 0.0;
                A[i][j] = delta - lambda * q.weights[j] * K(q.points[i], q.points[j]);
            }
            B[i] = f(q.points[i]);
        }

        std::vector<double> phi_nodes;
        if (!solveLinearSystem(A, B, phi_nodes)) return {};
        return phi_nodes;
    }

    static double interpolate(double x, double a, double b, double lambda,
                              KernelFunc K, SourceFunc f,
                              const std::vector<double>& phi_nodes,
                              int n = 8) {
        Quadrature q = Quadrature::GaussLegendreN(n, a, b);
        double integral = 0;
        for (size_t j = 0; j < q.points.size(); ++j) {
            integral += q.weights[j] * K(x, q.points[j]) * phi_nodes[j];
        }
        return f(x) + lambda * integral;
    }

    // Helper for Fredholm equation of the 1st kind (via Tikhonov Regularization):
    // integral K(x,y) phi(y) dy = f(x)
    // Minimizing ||K phi - f||^2 + alpha ||phi||^2
    // Leads to: alpha*phi(x) + integral L(x,y) phi(y) dy = g(x)
    // where L(x,y) = integral K(z,x) K(z,y) dz and g(x) = integral K(z,x) f(z) dz
};

template<typename T>
class AdaptiveCompensator {
private:
    struct Params {
        T kernelWidth = T(0.1);
        T lambda = T(0.9);
    } params;

    std::vector<T> history;
    size_t maxHistory = 100;

public:
    void setParams(T width, T lambda) {
        params.kernelWidth = width;
        params.lambda = lambda;
    }

    T compensate(T input) {
        history.push_back(input);
        if (history.size() > maxHistory) history.erase(history.begin());

        auto K = [this](T x, T y) {
            T diff = x - y;
            return std::exp(-diff * diff / (2 * params.kernelWidth * params.kernelWidth));
        };

        T integral = 0;
        T totalWeight = 0;
        for (T h : history) {
            T w = K(input, h);
            integral += w * h;
            totalWeight += w;
        }

        if (totalWeight < 1e-9) return input;
        T smoothed = integral / totalWeight;
        return input * (1.0 - params.lambda) + smoothed * params.lambda;
    }
};

} // namespace Fredholm

#endif
