#ifndef RNN_ABSOLUTE_CORE_H
#define RNN_ABSOLUTE_CORE_H

#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <random>
#include <algorithm>
#include <fstream>

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

class MultiDimRNN {
public:
    MultiDimRNN(int in, int hid, int out) : input_size(in), hidden_size(hid), output_size(out) {
        weights_ih.resize(in * hid);
        weights_hh.resize(hid * hid);
        weights_ho.resize(hid * out);
        bias_h.resize(hid, 0.0);
        bias_o.resize(out, 0.0);

        std::default_random_engine gen(42);
        std::uniform_real_distribution<double> dist(-0.5 / std::sqrt(hid), 0.5 / std::sqrt(hid));
        for(auto& w : weights_ih) w = dist(gen);
        for(auto& w : weights_hh) w = dist(gen);
        for(auto& w : weights_ho) w = dist(gen);

        h_next.resize(hid);
        out_buf.resize(out);
        out_err.resize(out);
        h_err.resize(hid);
    }

    const std::vector<double>& forward(const std::vector<double>& input, std::vector<double>& h_state) {
        for (int i = 0; i < hidden_size; ++i) {
            double sum = bias_h[i];
            for (int j = 0; j < input_size; ++j) sum += input[j] * weights_ih[j * hidden_size + i];
            for (int j = 0; j < hidden_size; ++j) sum += h_state[j] * weights_hh[j * hidden_size + i];
            h_next[i] = std::tanh(sum);
        }
        h_state = h_next;
        for (int i = 0; i < output_size; ++i) {
            double sum = bias_o[i];
            for (int j = 0; j < hidden_size; ++j) sum += h_state[j] * weights_ho[j * output_size + i];
            out_buf[i] = sum;
        }
        return out_buf;
    }

    void trainStep(const std::vector<double>& input, const std::vector<double>& target, std::vector<double>& h_state, double lr) {
        std::vector<double> prev_h = h_state;
        const std::vector<double>& pred = forward(input, h_state);

        for (int i = 0; i < output_size; ++i) out_err[i] = pred[i] - target[i];

        for (int i = 0; i < output_size; ++i) {
            bias_o[i] -= lr * out_err[i];
            for (int j = 0; j < hidden_size; ++j) weights_ho[j * output_size + i] -= lr * out_err[i] * h_state[j];
        }

        std::fill(h_err.begin(), h_err.end(), 0.0);
        for (int j = 0; j < hidden_size; ++j) {
            for (int i = 0; i < output_size; ++i) h_err[j] += out_err[i] * weights_ho[j * output_size + i];
            h_err[j] *= (1.0 - h_state[j] * h_state[j]);
        }

        for (int i = 0; i < hidden_size; ++i) {
            bias_h[i] -= lr * h_err[i];
            for (int j = 0; j < input_size; ++j) weights_ih[j * hidden_size + i] -= lr * h_err[i] * input[j];
            for (int j = 0; j < hidden_size; ++j) weights_hh[j * hidden_size + i] -= lr * h_err[i] * prev_h[j];
        }
    }

private:
    int input_size, hidden_size, output_size;
    std::vector<double> weights_ih, weights_hh, weights_ho;
    std::vector<double> bias_h, bias_o;
    std::vector<double> h_next, out_buf, out_err, h_err;
};

#endif
