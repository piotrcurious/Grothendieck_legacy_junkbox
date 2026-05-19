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
    FiniteFieldElement operator*(const FiniteFieldElement& other) const { return FiniteFieldElement((value * other.value) % PRIME); }
    FiniteFieldElement inverse() const { return pow(PRIME - 2); }
    FiniteFieldElement pow(int exp) const {
        if (exp == 0) return FiniteFieldElement(1);
        FiniteFieldElement half = pow(exp / 2);
        FiniteFieldElement result = half * half;
        if (exp % 2 != 0) result = result * (*this);
        return result;
    }
    friend std::ostream& operator<<(std::ostream& os, const FiniteFieldElement& elem) {
        os << elem.value;
        return os;
    }
};

class FiniteFieldVector {
public:
    std::vector<FiniteFieldElement> elements;
    FiniteFieldVector(const std::vector<int>& values) {
        for (int val : values) elements.emplace_back(val);
    }
    FiniteFieldVector(const std::vector<FiniteFieldElement>& elems) : elements(elems) {}
    friend std::ostream& operator<<(std::ostream& os, const FiniteFieldVector& vec) {
        for (const auto& elem : vec.elements) os << elem << " ";
        return os;
    }
};

class MultiDimRNN {
public:
    MultiDimRNN(int in, int hid, int out) : input_size(in), hidden_size(hid), output_size(out) {
        weights_ih.resize(in * hid);
        weights_hh.resize(hid * hid);
        weights_ho.resize(hid * out);
        std::generate(weights_ih.begin(), weights_ih.end(), random_init);
        std::generate(weights_hh.begin(), weights_hh.end(), random_init);
        std::generate(weights_ho.begin(), weights_ho.end(), random_init);
    }

    std::vector<double> forward(const std::vector<double>& input, std::vector<double>& h_state) {
        std::vector<double> next_h(hidden_size, 0.0);
        for (int i = 0; i < hidden_size; ++i) {
            for (int j = 0; j < input_size; ++j) next_h[i] += input[j] * weights_ih[j * hidden_size + i];
            for (int j = 0; j < hidden_size; ++j) next_h[i] += h_state[j] * weights_hh[j * hidden_size + i];
            next_h[i] = std::tanh(next_h[i]);
        }
        h_state = next_h;
        std::vector<double> output(output_size, 0.0);
        for (int i = 0; i < output_size; ++i) {
            for (int j = 0; j < hidden_size; ++j) output[i] += h_state[j] * weights_ho[j * output_size + i];
        }
        return output;
    }

    void train(const std::vector<std::vector<double>>& sequences, int epochs, double lr) {
        for (int e = 0; e < epochs; ++e) {
            double total_loss = 0;
            for (size_t s = 0; s < sequences.size() - 1; ++s) {
                std::vector<double> h(hidden_size, 0.0);
                std::vector<double> pred = forward(sequences[s], h);
                const std::vector<double>& target = sequences[s+1];
                for (int i = 0; i < output_size; ++i) {
                    double error = pred[i] - target[i];
                    total_loss += error * error;
                    for (int j = 0; j < hidden_size; ++j) {
                        weights_ho[j * output_size + i] -= lr * error * h[j];
                    }
                }
            }
            if (e % 10 == 0) std::cout << "Epoch " << e << " Loss: " << total_loss << std::endl;
        }
    }

private:
    int input_size, hidden_size, output_size;
    std::vector<double> weights_ih, weights_hh, weights_ho;
    static double random_init() {
        static std::default_random_engine gen(42);
        static std::uniform_real_distribution<double> dist(-0.1, 0.1);
        return dist(gen);
    }
};

#endif
