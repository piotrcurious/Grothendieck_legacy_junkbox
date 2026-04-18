#ifndef FREDHOLM_ENGINE_H
#define FREDHOLM_ENGINE_H

#include <vector>
#include <cmath>
#include <functional>
#include <complex>
#include <algorithm>
#include <iostream>

namespace Fredholm {

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
        Quadrature q = GaussLegendre8();
        for (size_t i = 0; i < q.points.size(); ++i) {
            double p = q.points[i];
            q.points[i] = 0.5 * (b - a) * p + 0.5 * (a + b);
            q.weights[i] *= 0.5 * (b - a);
        }
        return q;
    }
};

// Matrix utilities for eigenvalue calculation
struct Matrix {
    int rows, cols;
    std::vector<double> data;

    Matrix(int r, int c) : rows(r), cols(c), data(r * c, 0.0) {}

    double& operator()(int r, int c) { return data[r * cols + c]; }
    double operator()(int r, int c) const { return data[r * cols + c]; }

    static Matrix Identity(int n) {
        Matrix res(n, n);
        for (int i = 0; i < n; i++) res(i, i) = 1.0;
        return res;
    }
};

inline void qrStep(Matrix& A, Matrix& Q_total) {
    int n = A.rows;
    Matrix Q = Matrix::Identity(n);
    Matrix R = A;

    for (int j = 0; j < n; j++) {
        for (int i = j + 1; i < n; i++) {
            if (std::abs(R(i, j)) > 1e-15) {
                double r = std::sqrt(R(j, j) * R(j, j) + R(i, j) * R(i, j));
                double c = R(j, j) / r;
                double s = -R(i, j) / r;

                for (int k = 0; k < n; k++) {
                    double r_jk = R(j, k);
                    double r_ik = R(i, k);
                    R(j, k) = c * r_jk - s * r_ik;
                    R(i, k) = s * r_jk + c * r_ik;

                    double q_kj = Q(k, j);
                    double q_ki = Q(k, i);
                    Q(k, j) = c * q_kj - s * q_ki;
                    Q(k, i) = s * q_kj + c * q_ki;
                }
            }
        }
    }
    // A_next = R * Q
    Matrix A_next(n, n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                A_next(i, j) += R(i, k) * Q(k, j);
            }
        }
    }
    A = A_next;

    // Accumulate Q: Q_total = Q_total * Q
    Matrix next_Q_total(n, n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                next_Q_total(i, j) += Q_total(i, k) * Q(k, j);
            }
        }
    }
    Q_total = next_Q_total;
}

inline void computeEigen(const Matrix& mat, std::vector<double>& eigenvalues, std::vector<std::vector<double>>& eigenvectors) {
    int n = mat.rows;
    Matrix A = mat;
    Matrix Q_total = Matrix::Identity(n);

    for (int iter = 0; iter < 100; iter++) {
        qrStep(A, Q_total);
    }

    eigenvalues.resize(n);
    for (int i = 0; i < n; i++) eigenvalues[i] = A(i, i);

    eigenvectors.resize(n, std::vector<double>(n));
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            eigenvectors[j][i] = Q_total(i, j);
        }
    }
}

enum class SolverStatus { SUCCESS, SINGULAR_MATRIX, NO_CONVERGENCE };

inline SolverStatus solveLinearSystem(std::vector<std::vector<double>>& A, std::vector<double>& b, std::vector<double>& x) {
    int n = b.size();
    double eps = 1e-18;
    for (int i = 0; i < n; i++) {
        double maxEl = std::abs(A[i][i]);
        int maxRow = i;
        for (int k = i + 1; k < n; k++) {
            if (std::abs(A[k][i]) > maxEl) {
                maxEl = std::abs(A[k][i]);
                maxRow = k;
            }
        }
        if (maxEl < eps) return SolverStatus::SINGULAR_MATRIX;
        std::swap(A[maxRow], A[i]);
        std::swap(b[maxRow], b[i]);
        for (int k = i + 1; k < n; k++) {
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
        x[i] = b[i] / A[i][i];
        for (int k = i - 1; k >= 0; k--) b[k] -= A[k][i] * x[i];
    }
    return SolverStatus::SUCCESS;
}

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
        if (solveLinearSystem(A, B, phi_nodes) != SolverStatus::SUCCESS) return {};
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
        T integral = 0; T totalWeight = 0;
        for (T h : history) {
            T w = K(input, h); integral += w * h; totalWeight += w;
        }
        if (totalWeight < 1e-9) return input;
        T smoothed = integral / totalWeight;
        return input * (1.0 - params.lambda) + smoothed * params.lambda;
    }
};

} // namespace Fredholm

#endif
