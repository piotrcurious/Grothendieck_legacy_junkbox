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
        std::uniform_real_distribution<double> dist(-1.0 / sqrt(in), 1.0 / sqrt(in));
        auto init = [&]() { return dist(gen); };
        std::generate(weights_ih.begin(), weights_ih.end(), init);
        std::generate(weights_hh.begin(), weights_hh.end(), init);
        std::generate(weights_ho.begin(), weights_ho.end(), init);
    }

    std::vector<double> forward(const std::vector<double>& input, std::vector<double>& h_state) {
        std::vector<double> next_h(hidden_size, 0.0);
        for (int i = 0; i < hidden_size; ++i) {
            double sum = bias_h[i];
            for (int j = 0; j < input_size; ++j) sum += input[j] * weights_ih[j * hidden_size + i];
            for (int j = 0; j < hidden_size; ++j) sum += h_state[j] * weights_hh[j * hidden_size + i];
            next_h[i] = std::tanh(sum);
        }
        h_state = next_h;
        std::vector<double> output(output_size, 0.0);
        for (int i = 0; i < output_size; ++i) {
            double sum = bias_o[i];
            for (int j = 0; j < hidden_size; ++j) sum += h_state[j] * weights_ho[j * output_size + i];
            output[i] = sum;
        }
        return output;
    }

    void trainStep(const std::vector<double>& input, const std::vector<double>& target, std::vector<double>& h_state, double lr) {
        std::vector<double> prev_h = h_state;
        std::vector<double> pred = forward(input, h_state);

        std::vector<double> out_error(output_size);
        for (int i = 0; i < output_size; ++i) out_error[i] = pred[i] - target[i];

        for (int i = 0; i < output_size; ++i) {
            bias_o[i] -= lr * out_error[i];
            for (int j = 0; j < hidden_size; ++j) weights_ho[j * output_size + i] -= lr * out_error[i] * h_state[j];
        }

        std::vector<double> h_error(hidden_size, 0.0);
        for (int j = 0; j < hidden_size; ++j) {
            for (int i = 0; i < output_size; ++i) h_error[j] += out_error[i] * weights_ho[j * output_size + i];
            h_error[j] *= (1.0 - h_state[j] * h_state[j]);
        }

        for (int i = 0; i < hidden_size; ++i) {
            bias_h[i] -= lr * h_error[i];
            for (int j = 0; j < input_size; ++j) weights_ih[j * hidden_size + i] -= lr * h_error[i] * input[j];
            for (int j = 0; j < hidden_size; ++j) weights_hh[j * hidden_size + i] -= lr * h_error[i] * prev_h[j];
        }
    }

private:
    int input_size, hidden_size, output_size;
    std::vector<double> weights_ih, weights_hh, weights_ho;
    std::vector<double> bias_h, bias_o;
};

#endif
