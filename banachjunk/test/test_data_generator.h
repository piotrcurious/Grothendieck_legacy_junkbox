#ifndef BANACHJUNK_TEST_DATA_GENERATOR_H
#define BANACHJUNK_TEST_DATA_GENERATOR_H

#include <vector>
#include <cmath>
#include <random>
#include "Arduino.h"

namespace banach {
namespace test {

struct SignalPoint {
    float t;
    float v;
};

class DataGenerator {
public:
    // Generates a noisy sinusoid with jittered sampling
    static std::vector<SignalPoint> generateNoisySine(int n, float freq, float amp, float noise_std, float avg_dt = 0.1f) {
        std::vector<SignalPoint> signal;
        float current_t = 0;
        std::mt19937 gen(42);
        std::normal_distribution<float> noise(0, noise_std);
        std::uniform_real_distribution<float> jitter(0.8f * avg_dt, 1.2f * avg_dt);

        for (int i = 0; i < n; ++i) {
            float val = amp * std::sin(2.0f * PI * freq * current_t) + noise(gen);
            signal.push_back({current_t, val});
            current_t += jitter(gen);
        }
        return signal;
    }

    // Generates a multi-segment signal (Linear + Quadratic + Constant)
    static std::vector<SignalPoint> generateComplexTrend(int n, float noise_std) {
        std::vector<SignalPoint> signal;
        std::mt19937 gen(1337);
        std::normal_distribution<float> noise(0, noise_std);
        float dt = 0.1f;

        for (int i = 0; i < n; ++i) {
            float t = i * dt;
            float val = 0;
            if (i < n / 3) {
                val = 2.0f * t; // Linear segment
            } else if (i < 2 * n / 3) {
                float t_rel = t - (n/3)*dt;
                val = (n/3)*dt*2.0f + 0.5f * t_rel * t_rel; // Quadratic segment
            } else {
                val = signal.back().v; // Constant plateau
            }
            signal.push_back({t, val + noise(gen)});
        }
        return signal;
    }

    // Generates a random walk with specific drift and volatility (Hurst proxy check)
    static std::vector<SignalPoint> generateRandomWalk(int n, float drift, float vol, float dt = 0.1f) {
        std::vector<SignalPoint> signal;
        std::mt19937 gen(777);
        std::normal_distribution<float> dist(0, 1.0f);
        float current_v = 1.0f;

        for (int i = 0; i < n; ++i) {
            signal.push_back({i * dt, current_v});
            current_v += drift * dt + vol * std::sqrt(dt) * dist(gen);
        }
        return signal;
    }

    // Generates a signal with a large spike at a specific location
    static std::vector<SignalPoint> generateSpikeSignal(int n, int spike_idx, float spike_amp) {
        std::vector<SignalPoint> signal;
        for (int i = 0; i < n; ++i) {
            float val = (i == spike_idx) ? spike_amp : (random(100) / 1000.0f);
            signal.push_back({i * 0.1f, val});
        }
        return signal;
    }

    // Generates a chirp signal (frequency modulation)
    static std::vector<SignalPoint> generateChirp(int n, float f0, float f1, float t1, float amp) {
        std::vector<SignalPoint> signal;
        float dt = t1 / n;
        float k = (f1 - f0) / t1;
        for (int i = 0; i < n; ++i) {
            float t = i * dt;
            float val = amp * std::sin(2.0f * PI * (f0 * t + 0.5f * k * t * t));
            signal.push_back({t, val});
        }
        return signal;
    }

    // Generates two signals with time-varying correlation
    static std::pair<std::vector<SignalPoint>, std::vector<SignalPoint>> generateCorrelatedSignals(int n, float noise_std) {
        std::vector<SignalPoint> s1, s2;
        std::mt19937 gen(42);
        std::normal_distribution<float> noise(0, noise_std);
        float dt = 0.1f;
        for (int i = 0; i < n; ++i) {
            float t = i * dt;
            float common = std::sin(t * 0.5f);
            // Correlation coefficient shifts from 0 to 1 back to 0
            float rho = std::sin(PI * i / n);
            float v1 = common + noise(gen);
            float v2 = rho * common + std::sqrt(1.0f - rho*rho) * noise(gen);
            s1.push_back({t, v1});
            s2.push_back({t, v2});
        }
        return {s1, s2};
    }

    // Generates a step function with exponential relaxation
    static std::vector<SignalPoint> generateStepRelaxation(int n, int step_idx, float amp, float tau) {
        std::vector<SignalPoint> signal;
        float dt = 0.1f;
        for (int i = 0; i < n; ++i) {
            float t = i * dt;
            float val = 0;
            if (i >= step_idx) {
                val = amp * (1.0f - std::exp(-(t - step_idx * dt) / tau));
            }
            signal.push_back({t, val});
        }
        return signal;
    }
};

} // namespace test
} // namespace banach

#endif // BANACHJUNK_TEST_DATA_GENERATOR_H
