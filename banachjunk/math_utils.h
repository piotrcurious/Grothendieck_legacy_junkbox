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
