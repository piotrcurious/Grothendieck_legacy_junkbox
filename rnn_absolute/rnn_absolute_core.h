#ifndef RNN_ABSOLUTE_CORE_H
#define RNN_ABSOLUTE_CORE_H

#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

const int PRIME = 10007;

class FiniteFieldElement {
public:
    int value;
    FiniteFieldElement(int v = 0) : value(v % PRIME) {
        if (value < 0) value += PRIME;
    }
    FiniteFieldElement operator+(const FiniteFieldElement& other) const { return FiniteFieldElement((value + other.value) % PRIME); }
    FiniteFieldElement operator-(const FiniteFieldElement& other) const { return FiniteFieldElement((value - other.value + PRIME) % PRIME); }
    FiniteFieldElement operator*(const FiniteFieldElement& other) const { return FiniteFieldElement((1LL * value * other.value) % PRIME); }
};

class OptimizedRNN {
public:
    OptimizedRNN(int in, int hid) : input_size(in), hidden_size(hid) {
        weights_ih.resize(in * hid);
        weights_hh.resize(hid * hid);
        weights_ho.resize(hid);
        bias_h.resize(hid, 0.0);
        bias_o = 0.0;

        m_ih.assign(in * hid, 0.0); v_ih.assign(in * hid, 0.0);
        m_hh.assign(hid * hid, 0.0); v_hh.assign(hid * hid, 0.0);
        m_ho.assign(hid, 0.0); v_ho.assign(hid, 0.0);
        m_bh.assign(hid, 0.0); v_bh.assign(hid, 0.0);

        std::default_random_engine gen(42);
        std::uniform_real_distribution<double> dist(-0.1, 0.1);
        for(auto& w : weights_ih) w = dist(gen);
        for(auto& w : weights_hh) w = dist(gen);
        for(auto& w : weights_ho) w = dist(gen);

        h_state.assign(hid, 0.0);
        h_prev.assign(hid, 0.0);
    }

    double forward(const std::vector<double>& input) {
        h_prev = h_state;
        for (int i = 0; i < hidden_size; ++i) {
            double sum = bias_h[i];
            for (int j = 0; j < input_size; ++j) sum += input[j] * weights_ih[j * hidden_size + i];
            for (int j = 0; j < hidden_size; ++j) sum += h_prev[j] * weights_hh[j * hidden_size + i];
            h_state[i] = std::tanh(sum);
        }
        double out = bias_o;
        for (int j = 0; j < hidden_size; ++j) out += h_state[j] * weights_ho[j];
        return out;
    }

    void train(const std::vector<double>& input, double target, double lr) {
        double out = bias_o;
        for (int j = 0; j < hidden_size; ++j) out += h_state[j] * weights_ho[j];

        double d_out = out - target;
        t++;
        const double b1 = 0.9, b2 = 0.999, eps = 1e-8;

        double bias_corr1 = 1.0 - std::pow(b1, t);
        double bias_corr2 = 1.0 - std::pow(b2, t);
        double step_size = lr / bias_corr1;

        auto update_fast = [&](double& w, double& m, double& v, double grad) {
            m = b1 * m + (1.0 - b1) * grad;
            v = b2 * v + (1.0 - b2) * grad * grad;
            w -= step_size * m / (std::sqrt(v / bias_corr2) + eps);
        };

        update_fast(bias_o, m_bo, v_bo, d_out);
        for(int j=0; j<hidden_size; ++j) {
            update_fast(weights_ho[j], m_ho[j], v_ho[j], d_out * h_state[j]);
            double d_h = d_out * weights_ho[j] * (1.0 - h_state[j] * h_state[j]);
            update_fast(bias_h[j], m_bh[j], v_bh[j], d_h);
            for(int k=0; k<input_size; ++k)
                update_fast(weights_ih[k*hidden_size + j], m_ih[k*hidden_size + j], v_ih[k*hidden_size + j], d_h * input[k]);
            for(int k=0; k<hidden_size; ++k)
                update_fast(weights_hh[k*hidden_size + j], m_hh[k*hidden_size + j], v_hh[k*hidden_size + j], d_h * h_prev[k]);
        }
    }

private:
    int input_size, hidden_size;
    std::vector<double> weights_ih, weights_hh, weights_ho, bias_h;
    double bias_o;
    std::vector<double> h_state, h_prev;
    std::vector<double> m_ih, v_ih, m_hh, v_hh, m_ho, v_ho, m_bh, v_bh;
    double m_bo = 0, v_bo = 0;
    long long t = 0;
};

#endif
