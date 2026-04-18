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
    static Quadrature GaussLegendre16() {
        return {
            {-0.9894009349916499, -0.9445750230732326, -0.8656312023878318, -0.7554044083550030,
             -0.6178762444026438, -0.4580167776572274, -0.2816035507792589, -0.0950125098376374,
              0.0950125098376374,  0.2816035507792589,  0.4580167776572274,  0.6178762444026438,
              0.7554044083550030,  0.8656312023878318,  0.9445750230732326,  0.9894009349916499},
            { 0.0271524594117541,  0.0622535239386479,  0.0951585116824928,  0.1246289712569356,
              0.1495959888165767,  0.1691565193950025,  0.1826034150449236,  0.1894506104550685,
              0.1894506104550685,  0.1826034150449236,  0.1691565193950025,  0.1495959888165767,
              0.1246289712569356,  0.0951585116824928,  0.0622535239386479,  0.0271524594117541}
        };
    }
    static Quadrature GaussLegendreN(int n, double a, double b) {
        Quadrature q = (n > 8) ? GaussLegendre16() : GaussLegendre8();
        for (size_t i = 0; i < q.points.size(); ++i) {
            double p = q.points[i];
            q.points[i] = 0.5 * (b - a) * p + 0.5 * (a + b);
            q.weights[i] *= 0.5 * (b - a);
        }
        return q;
    }
};

struct Matrix {
    int rows, cols;
    std::vector<double> data;
    Matrix(int r, int c) : rows(r), cols(c), data(r * c, 0.0) {}
    double& operator()(int r, int c) { return data[r * cols + c]; }
    double operator()(int r, int c) const { return data[r * cols + c]; }
    static Matrix Identity(int n) { Matrix res(n, n); for (int i = 0; i < n; i++) res(i, i) = 1.0; return res; }
    Matrix transpose() const {
        Matrix res(cols, rows);
        for(int i=0; i<rows; i++) for(int j=0; j<cols; j++) res(j, i) = (*this)(i, j);
        return res;
    }
    Matrix operator*(const Matrix& other) const {
        Matrix res(rows, other.cols);
        for(int i=0; i<rows; i++) for(int k=0; k<cols; k++) {
            double val = (*this)(i, k);
            for(int j=0; j<other.cols; j++) res(i, j) += val * other(k, j);
        }
        return res;
    }
};

inline void qrStep(Matrix& A, Matrix& Q_total) {
    int n = A.rows; Matrix Q = Matrix::Identity(n); Matrix R = A;
    for (int j = 0; j < n; j++) {
        for (int i = j + 1; i < n; i++) {
            if (std::abs(R(i, j)) > 1e-15) {
                double r = std::sqrt(R(j, j) * R(j, j) + R(i, j) * R(i, j));
                double c = R(j, j) / r; double s = -R(i, j) / r;
                for (int k = 0; k < n; k++) {
                    double r_jk = R(j, k); double r_ik = R(i, k);
                    R(j, k) = c * r_jk - s * r_ik; R(i, k) = s * r_jk + c * r_ik;
                    double q_kj = Q(k, j); double q_ki = Q(k, i);
                    Q(k, j) = c * q_kj - s * q_ki; Q(k, i) = s * q_kj + c * q_ki;
                }
            }
        }
    }
    Matrix A_next(n, n);
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) for (int k = 0; k < n; k++) A_next(i, j) += R(i, k) * Q(k, j);
    A = A_next;
    Matrix next_Q_total(n, n);
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) for (int k = 0; k < n; k++) next_Q_total(i, j) += Q_total(i, k) * Q(k, j);
    Q_total = next_Q_total;
}

inline void computeEigen(const Matrix& mat, std::vector<double>& eigenvalues, std::vector<std::vector<double>>& eigenvectors) {
    int n = mat.rows; Matrix A = mat; Matrix Q_total = Matrix::Identity(n);
    for (int iter = 0; iter < 100; iter++) qrStep(A, Q_total);
    eigenvalues.resize(n); for (int i = 0; i < n; i++) eigenvalues[i] = A(i, i);
    eigenvectors.resize(n, std::vector<double>(n));
    for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) eigenvectors[j][i] = Q_total(i, j);
}

inline double powerIteration(const Matrix& A, int iters = 50) {
    int n = A.rows; std::vector<double> v(n, 1.0);
    double lambda = 0;
    for(int it=0; it<iters; it++) {
        std::vector<double> v_next(n, 0.0);
        for(int i=0; i<n; i++) for(int j=0; j<n; j++) v_next[i] += A(i, j) * v[j];
        double norm = 0; for(double x : v_next) norm += x*x; norm = std::sqrt(norm);
        if(norm < 1e-18) return 0;
        for(int i=0; i<n; i++) v_next[i] /= norm;
        double rayleigh = 0;
        for(int i=0; i<n; i++) {
            double Av_i = 0; for(int j=0; j<n; j++) Av_i += A(i, j) * v_next[j];
            rayleigh += v_next[i] * Av_i;
        }
        lambda = rayleigh; v = v_next;
    }
    return std::abs(lambda);
}

enum class SolverStatus { SUCCESS, SINGULAR_MATRIX, NO_CONVERGENCE };

inline SolverStatus solveLinearSystem(std::vector<std::vector<double>>& A, std::vector<double>& b, std::vector<double>& x) {
    int n = b.size(); double eps = 1e-18;
    for (int i = 0; i < n; i++) {
        double maxEl = std::abs(A[i][i]); int maxRow = i;
        for (int k = i + 1; k < n; k++) if (std::abs(A[k][i]) > maxEl) { maxEl = std::abs(A[k][i]); maxRow = k; }
        if (maxEl < eps) return SolverStatus::SINGULAR_MATRIX;
        std::swap(A[maxRow], A[i]);
        std::swap(b[maxRow], b[i]);
        for (int k = i + 1; k < n; k++) {
            double c = -A[k][i] / A[i][i];
            for (int j = i; j < n; j++) if (i == j) A[k][j] = 0; else A[k][j] += c * A[i][j];
            b[k] += c * b[i];
        }
    }
    x.resize(n);
    for (int i = n - 1; i >= 0; i--) { x[i] = b[i] / A[i][i]; for (int k = i - 1; k >= 0; k--) b[k] -= A[k][i] * x[i]; }
    return SolverStatus::SUCCESS;
}

class Solver {
public:
    using KernelFunc = std::function<double(double, double)>;
    using SourceFunc = std::function<double(double)>;

    static std::vector<double> solveFredholm(double a, double b, double lambda, KernelFunc K, SourceFunc f, int n = 16) {
        Quadrature q = Quadrature::GaussLegendreN(n, a, b); int N = q.points.size();
        std::vector<std::vector<double>> A(N, std::vector<double>(N)); std::vector<double> B(N);
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

    static double estimateConditionNumber(double a, double b, double lambda, KernelFunc K, int n = 16) {
        Quadrature q = Quadrature::GaussLegendreN(n, a, b); int N = q.points.size();
        Matrix A(N, N);
        for (int i = 0; i < N; ++i) for (int j = 0; j < N; ++j) {
            double delta = (i == j) ? 1.0 : 0.0;
            A(i, j) = delta - lambda * q.weights[j] * K(q.points[i], q.points[j]);
        }
        Matrix At = A.transpose();
        Matrix AtA = At * A;
        double max_sing_sq = powerIteration(AtA);
        return std::sqrt(max_sing_sq);
    }

    static std::vector<double> solveVolterra(double a, double b, double lambda, KernelFunc K, SourceFunc f, int n_steps = 100) {
        double h = (b - a) / n_steps; std::vector<double> x(n_steps + 1); std::vector<double> phi(n_steps + 1);
        for (int i = 0; i <= n_steps; i++) x[i] = a + i * h;
        phi[0] = f(x[0]);
        for (int i = 1; i <= n_steps; i++) {
            double sum = 0;
            for (int j = 0; j < i; j++) { double term = K(x[i], x[j]) * phi[j]; if (j == 0) sum += 0.5 * term; else sum += term; }
            double denom = 1.0 - 0.5 * lambda * h * K(x[i], x[i]);
            if (std::abs(denom) < 1e-12) return {};
            phi[i] = (f(x[i]) + lambda * h * sum) / denom;
        }
        return phi;
    }

    static double interpolateFredholm(double x, double a, double b, double lambda, KernelFunc K, SourceFunc f, const std::vector<double>& phi_nodes, int n = 16) {
        Quadrature q = Quadrature::GaussLegendreN(n, a, b); double integral = 0;
        for (size_t j = 0; j < q.points.size(); ++j) integral += q.weights[j] * K(x, q.points[j]) * phi_nodes[j];
        return f(x) + lambda * integral;
    }

    static std::vector<double> neumannStep(const std::vector<double>& phi_prev, double a, double b, double lambda, KernelFunc K, SourceFunc f, int n = 16) {
        Quadrature q = Quadrature::GaussLegendreN(n, a, b); int N = q.points.size();
        std::vector<double> phi_next(N);
        for(int i=0; i<N; i++) {
            double integral = 0;
            for(int j=0; j<N; j++) integral += q.weights[j] * K(q.points[i], q.points[j]) * phi_prev[j];
            phi_next[i] = f(q.points[i]) + lambda * integral;
        }
        return phi_next;
    }

    static double legendreP(int n, double x) {
        if(n == 0) return 1.0;
        if(n == 1) return x;
        double p0 = 1.0, p1 = x, pn = 0;
        for(int i=2; i<=n; i++) {
            pn = ((2.0*i - 1.0)*x*p1 - (i - 1.0)*p0) / (double)i;
            p0 = p1; p1 = pn;
        }
        return pn;
    }

    // Optimized Galerkin matrix construction
    static std::vector<double> solveGalerkinOptimized(double a, double b, double lambda, KernelFunc K, SourceFunc f, int degree) {
        int N = degree + 1;
        std::vector<std::vector<double>> A(N, std::vector<double>(N));
        std::vector<double> B(N);
        Quadrature q = Quadrature::GaussLegendre16();

        // Pre-evaluate basis functions at quadrature points
        std::vector<std::vector<double>> basis_evals(N, std::vector<double>(16));
        for(int i=0; i<N; i++) {
            for(int k=0; k<16; k++) {
                basis_evals[i][k] = legendreP(i, q.points[k]);
            }
        }

        for(int i=0; i<N; i++) {
            for(int j=0; j<N; j++) {
                double integral = 0;
                for(int n1=0; n1<16; n1++) {
                    double x_val = 0.5 * (b - a) * q.points[n1] + 0.5 * (a + b);
                    double w1_scaled = q.weights[n1] * 0.5 * (b - a);
                    for(int n2=0; n2<16; n2++) {
                        double y_val = 0.5 * (b - a) * q.points[n2] + 0.5 * (a + b);
                        double w2_scaled = q.weights[n2] * 0.5 * (b - a);
                        integral += w1_scaled * w2_scaled * basis_evals[i][n1] * K(x_val, y_val) * basis_evals[j][n2];
                    }
                }
                // Integral of P_i * P_j is 2/(2i+1) on [-1, 1]
                // We are on [a, b], so we need to scale the mass matrix accordingly
                double orth = (i == j) ? (b - a) / (2.0 * i + 1.0) : 0.0;
                A[i][j] = orth - lambda * integral;
            }
            double b_int = 0;
            for(int n=0; n<16; n++) {
                double x_val = 0.5 * (b - a) * q.points[n] + 0.5 * (a + b);
                double w_scaled = q.weights[n] * 0.5 * (b - a);
                b_int += w_scaled * f(x_val) * basis_evals[i][n];
            }
            B[i] = b_int;
        }
        std::vector<double> coeffs;
        if(solveLinearSystem(A, B, coeffs) != SolverStatus::SUCCESS) return {};
        return coeffs;
    }
};

template<typename T>
class AdaptiveCompensator {
private:
    struct Params { T kernelWidth = T(0.1); T lambda = T(0.9); } params;
    std::vector<T> history; size_t maxHistory = 100;
public:
    void setParams(T width, T lambda) { params.kernelWidth = width; params.lambda = lambda; }
    T compensate(T input) {
        history.push_back(input); if (history.size() > maxHistory) history.erase(history.begin());
        auto K = [this](T x, T y) { T diff = x - y; return std::exp(-diff * diff / (2 * params.kernelWidth * params.kernelWidth)); };
        T integral = 0; T totalWeight = 0;
        for (T h : history) { T w = K(input, h); integral += w * h; totalWeight += w; }
        if (totalWeight < 1e-9) return input;
        T smoothed = integral / totalWeight;
        return input * (1.0 - params.lambda) + smoothed * params.lambda;
    }
};

} // namespace Fredholm

#endif
