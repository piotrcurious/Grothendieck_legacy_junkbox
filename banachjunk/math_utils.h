#ifndef BANACHJUNK_MATH_UTILS_H
#define BANACHJUNK_MATH_UTILS_H

#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <optional>

namespace banach {

// Lebesgue-weighted statistics
class Statistics {
public:
    template<typename T>
    static T calculateMean(const std::vector<T>& values, const std::vector<T>& timestamps) {
        if (values.empty() || timestamps.size() < 2) return 0;
        T integral = 0;
        T totalDt = timestamps.back() - timestamps.front();
        if (std::abs(totalDt) < 1e-9) return values[0];

        for (size_t i = 0; i < values.size() - 1; i++) {
            T dt = timestamps[i + 1] - timestamps[i];
            integral += (values[i] + values[i + 1]) / 2.0 * dt;
        }
        return integral / totalDt;
    }

    template<typename T>
    static T calculateVariance(const std::vector<T>& values, const std::vector<T>& timestamps) {
        if (values.size() < 2 || timestamps.size() < 2) return 0;
        T mean = calculateMean(values, timestamps);
        T integral = 0;
        T totalDt = timestamps.back() - timestamps.front();
        if (std::abs(totalDt) < 1e-9) return 0;

        for (size_t i = 0; i < values.size() - 1; i++) {
            T dt = timestamps[i + 1] - timestamps[i];
            T d1 = values[i] - mean;
            T d2 = values[i + 1] - mean;
            integral += (d1 * d1 + d2 * d2) / 2.0 * dt;
        }
        return integral / totalDt;
    }

    struct Moments {
        double mean = 0;
        double variance = 0;
        double skewness = 0;
        double kurtosis = 0;
    };

    template<typename T>
    static Moments calculateMoments(const std::vector<T>& values, const std::vector<T>& timestamps) {
        Moments m;
        if (values.size() < 2 || timestamps.size() < 2) return m;
        double totalMeasure = timestamps.back() - timestamps.front();
        if (std::abs(totalMeasure) < 1e-9) return m;

        m.mean = calculateMean(values, timestamps);
        double varI = 0, m3I = 0, m4I = 0;
        for (size_t i = 0; i < values.size() - 1; ++i) {
            double dt = timestamps[i+1] - timestamps[i];
            double d1 = values[i] - m.mean;
            double d2 = values[i+1] - m.mean;
            double midD = (d1 + d2) / 2.0;
            varI += midD * midD * dt;
            m3I += midD * midD * midD * dt;
            m4I += midD * midD * midD * midD * dt;
        }
        m.variance = varI / totalMeasure;
        double sdev = std::sqrt(m.variance);
        if (sdev > 1e-9) {
            m.skewness = (m3I / totalMeasure) / (sdev * sdev * sdev);
            m.kurtosis = (m4I / totalMeasure) / (m.variance * m.variance) - 3.0;
        }
        return m;
    }

    template<typename T>
    static float calculateFlatness(const std::vector<T>& values, const std::vector<T>& timestamps) {
        if (values.size() < 2 || timestamps.size() < 2) return 0;
        double totalMeasure = timestamps.back() - timestamps.front();
        if (std::abs(totalMeasure) < 1e-9) return 0;

        double logSum = 0, arithSum = 0;
        for (size_t i = 0; i < values.size() - 1; ++i) {
            double dt = timestamps[i+1] - timestamps[i];
            double val = (std::abs((double)values[i]) + std::abs((double)values[i+1])) / 2.0;
            if (val < 1e-9) val = 1e-9;
            logSum += std::log(val) * dt;
            arithSum += val * dt;
        }
        double geomMean = std::exp(logSum / totalMeasure);
        double arithMean = arithSum / totalMeasure;
        return (arithMean > 1e-9) ? (float)(geomMean / arithMean) : 0.0f;
    }

    template<typename T>
    static float calculateHurst(const std::vector<T>& values, const std::vector<T>& timestamps) {
        if (values.size() < 8) return 0.5f;
        double mean = calculateMean(values, timestamps);
        double cumSum = 0, minZ = 1e30, maxZ = -1e30, sqSum = 0;
        for (const auto& val : values) {
            cumSum += (val - mean);
            if (cumSum < minZ) minZ = cumSum;
            if (cumSum > maxZ) maxZ = cumSum;
            sqSum += (val - mean) * (val - mean);
        }
        double sdev = std::sqrt(sqSum / values.size());
        if (sdev < 1e-9) return 0.5f;
        double rs = (maxZ - minZ) / sdev;
        if (rs <= 0) return 0.5f;
        return (float)(std::log(rs) / std::log((double)values.size()));
    }

    template<typename T>
    static double calculateLpNorm(const std::vector<std::vector<T>>& dimData, const std::vector<T>& timestamps, int p) {
        if (dimData.empty() || timestamps.size() < 2) return 0;
        double totalMeasure = timestamps.back() - timestamps.front();
        if (std::abs(totalMeasure) < 1e-9) return 0;

        double acc = 0;
        for (const auto& dimension : dimData) {
            if (dimension.size() < 2) continue;
            double maxV = 0;
            for (const auto& v : dimension) maxV = std::max(maxV, std::abs((double)v));
            if (maxV < 1e-9) continue;

            double integral = 0;
            for (size_t i = 0; i < dimension.size() - 1; ++i) {
                double dt = timestamps[i+1] - timestamps[i];
                double val = (std::abs((double)dimension[i]) + std::abs((double)dimension[i+1])) / (2.0 * maxV);
                integral += std::pow(val, p) * dt;
            }
            acc += maxV * std::pow(integral / totalMeasure, 1.0/p);
        }
        return acc / dimData.size();
    }

    template<typename T>
    static float calculateApEn(const std::vector<T>& data, int m = 2) {
        if (data.size() < (size_t)m + 1) return 0;

        auto phi = [&](int _m) {
            int N = data.size();
            // Estimate threshold r = 0.2 * sdev
            double sum = 0, sqSum = 0;
            for(auto v : data) { sum += v; sqSum += v*v; }
            double mean = sum / N;
            double sdev = std::sqrt(std::max(0.0, sqSum / N - mean * mean));
            double r = 0.2 * sdev;
            if (r < 1e-9) r = 0.01;

            double sumLogC = 0;
            for (int i = 0; i <= N - _m; ++i) {
                int count = 0;
                for (int j = 0; j <= N - _m; ++j) {
                    bool match = true;
                    for (int k = 0; k < _m; ++k) {
                        if (std::abs((double)data[i+k] - (double)data[j+k]) > r) {
                            match = false;
                            break;
                        }
                    }
                    if (match) count++;
                }
                sumLogC += std::log((double)count / (N - _m + 1));
            }
            return sumLogC / (N - _m + 1);
        };
        return (float)std::abs(phi(m) - phi(m + 1));
    }

    template<typename T>
    static float calculateCoherence(const std::vector<T>& d1, const std::vector<T>& d2, size_t start, size_t len) {
        if (d1.size() < start + len || d2.size() < start + len || len < 2) return 0;
        double m1 = 0, m2 = 0;
        for (size_t k = start; k < start + len; ++k) {
            m1 += d1[k]; m2 += d2[k];
        }
        m1 /= len; m2 /= len;

        double num = 0, den1 = 0, den2 = 0;
        for (size_t k = start; k < start + len; ++k) {
            double v1 = d1[k] - m1;
            double v2 = d2[k] - m2;
            num += v1 * v2;
            den1 += v1 * v1;
            den2 += v2 * v2;
        }
        return (den1 * den2 > 1e-9) ? (float)(num / std::sqrt(den1 * den2)) : 0.0f;
    }
};

// Legendre Polynomials (shifted to [t_min, t_max])
class LegendreBasis {
public:
    static float P(int n, float t, float t_min, float t_max) {
        if (t_max <= t_min) return 0;
        float x = (2.0f * t - (t_max + t_min)) / (t_max - t_min);
        if (n == 0) return 1.0f;
        if (n == 1) return x;
        if (n == 2) return 0.5f * (3.0f * x * x - 1.0f);
        if (n == 3) return 0.5f * (5.0f * x * x * x - 3.0f * x);
        return 0;
    }

    static std::vector<float> project(const std::vector<float>& data, const std::vector<float>& timestamps, int max_degree = 3) {
        if (data.size() < 2 || timestamps.size() < 2) return {};
        float t_min = timestamps.front();
        float t_max = timestamps.back();
        std::vector<float> coeffs(max_degree + 1, 0);

        for (int n = 0; n <= max_degree; ++n) {
            float integral = 0;
            float norm_sq = 0;
            for (size_t i = 0; i < data.size(); ++i) {
                float dt = 1.0;
                if (data.size() > 1) {
                    if (i == 0) dt = timestamps[1] - timestamps[0];
                    else if (i == data.size() - 1) dt = timestamps[i] - timestamps[i-1];
                    else dt = (timestamps[i+1] - timestamps[i-1]) / 2.0;
                }
                float pn = P(n, timestamps[i], t_min, t_max);
                integral += dt * data[i] * pn;
                norm_sq += dt * pn * pn;
            }
            if (norm_sq > 1e-9) coeffs[n] = integral / norm_sq;
        }
        return coeffs;
    }
};

// Ridge Regression for Polynomial Fitting
class LinearSolvers {
public:
    template<int M>
    struct RegressionResult {
        std::array<double, M> coefficients;
        float conditionProxy;
        bool success;
    };

    // Solves Ax = B using Gaussian elimination with partial pivoting and Ridge regularization
    template<int M>
    static RegressionResult<M> solveRidge(const std::vector<double>& x, const std::vector<double>& y,
                                        const std::vector<double>& dt, double lambda = 1e-4) {
        RegressionResult<M> res;
        res.success = false;
        double matrix[M][M] = {0};
        double rhs[M] = {0};

        for (size_t i = 0; i < x.size(); ++i) {
            double xp[2 * M];
            xp[0] = 1.0;
            for (int p = 1; p < 2 * M; ++p) xp[p] = xp[p - 1] * x[i];

            for (int r = 0; r < M; ++r) {
                for (int c = 0; c < M; ++c) {
                    matrix[r][c] += dt[i] * xp[r + c];
                }
                rhs[r] += dt[i] * y[i] * xp[r];
            }
        }

        for (int i = 0; i < M; i++) matrix[i][i] += lambda;

        // Gaussian elimination
        for (int i = 0; i < M; i++) {
            int pivot = i;
            for (int j = i + 1; j < M; j++) {
                if (std::abs(matrix[j][i]) > std::abs(matrix[pivot][i])) pivot = j;
            }
            for (int k = i; k < M; k++) std::swap(matrix[i][k], matrix[pivot][k]);
            std::swap(rhs[i], rhs[pivot]);

            if (std::abs(matrix[i][i]) < 1e-12) continue;

            for (int j = i + 1; j < M; j++) {
                double factor = matrix[j][i] / matrix[i][i];
                rhs[j] -= factor * rhs[i];
                for (int k = i; k < M; k++) matrix[j][k] -= factor * matrix[i][k];
            }
        }

        double d_min = 1e30, d_max = -1e30;
        for (int i = 0; i < M; i++) {
            double abs_d = std::abs(matrix[i][i]);
            if (abs_d < d_min) d_min = abs_d;
            if (abs_d > d_max) d_max = abs_d;
        }
        res.conditionProxy = (d_min > 1e-15) ? (float)(d_max / d_min) : 1e15f;

        for (int i = M - 1; i >= 0; i--) {
            if (std::abs(matrix[i][i]) < 1e-15) {
                res.coefficients[i] = 0;
            } else {
                double sum = 0;
                for (int j = i + 1; j < M; j++) sum += matrix[i][j] * res.coefficients[j];
                res.coefficients[i] = (rhs[i] - sum) / matrix[i][i];
            }
        }
        res.success = true;
        return res;
    }
};

// Circular buffer for real-time processing
template<typename T, size_t Size>
class CircularBuffer {
private:
    std::array<T, Size> buffer;
    size_t head = 0;
    size_t tail = 0;
    bool is_full = false;

public:
    void push(const T& item) {
        buffer[head] = item;
        head = (head + 1) % Size;
        if (is_full) {
            tail = (tail + 1) % Size;
        }
        if (head == tail) {
            is_full = true;
        }
    }

    std::optional<T> pop() {
        if (empty()) return std::nullopt;
        T item = buffer[tail];
        tail = (tail + 1) % Size;
        is_full = false;
        return item;
    }

    bool empty() const { return !is_full && (head == tail); }
    bool full() const { return is_full; }
    size_t size() const {
        if (is_full) return Size;
        if (head >= tail) return head - tail;
        return Size + head - tail;
    }

    void clear() {
        head = 0;
        tail = 0;
        is_full = false;
    }
};

} // namespace banach

#endif // BANACHJUNK_MATH_UTILS_H
