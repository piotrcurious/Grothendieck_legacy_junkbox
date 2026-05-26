#ifndef RNN_ABSOLUTE_CORE_H
#define RNN_ABSOLUTE_CORE_H

#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <cstdint>
#include <lzma.h>
#include <map>
#include <set>
#include <iostream>

// --- Rigorous Galois Field GF(2^8) ---
class GaloisField8 {
public:
    static const uint16_t POLY = 0x11B;
    uint8_t mul_table[256][256];
    uint8_t exp_table[256];
    uint8_t log_table[256];
    uint8_t inv_table[256];

    struct Orbit {
        int id;
        std::vector<uint8_t> elements;
        int degree;
    };

    std::vector<Orbit> orbits;
    int element_to_orbit_id[256];
    int element_to_root_index[256];

    GaloisField8() {
        uint16_t x = 1;
        for (int i = 0; i < 255; i++) {
            exp_table[i] = (uint8_t)x; log_table[x] = (uint8_t)i;
            x <<= 1; if (x & 0x100) x ^= POLY;
        }
        exp_table[255] = exp_table[0]; log_table[0] = 0;
        for (int i = 0; i < 256; i++) for (int j = 0; j < 256; j++) mul_table[i][j] = compute_mul(i, j);
        inv_table[0] = 0;
        for (int i = 1; i < 256; i++) for (int j = 1; j < 256; j++) if (mul_table[i][j] == 1) { inv_table[i] = (uint8_t)j; break; }

        bool visited[256] = {false};
        for (int i = 0; i < 256; i++) {
            if (visited[i]) continue;
            Orbit orb; orb.id = (int)orbits.size();
            uint8_t cur = (uint8_t)i;
            while (!visited[cur]) {
                visited[cur] = true;
                orb.elements.push_back(cur);
                cur = mul_table[cur][cur];
            }
            orb.degree = (int)orb.elements.size();
            std::sort(orb.elements.begin(), orb.elements.end());
            for (size_t j = 0; j < orb.elements.size(); j++) {
                element_to_orbit_id[orb.elements[j]] = orb.id;
                element_to_root_index[orb.elements[j]] = (int)j;
            }
            orbits.push_back(orb);
        }
    }

    uint8_t compute_mul(uint8_t a, uint8_t b) {
        uint8_t res = 0;
        for (int i = 0; i < 8; i++) {
            if (b & 1) res ^= a;
            bool hi = (a & 0x80); a <<= 1; if (hi) a ^= (POLY & 0xFF);
            b >>= 1;
        }
        return res;
    }

    inline uint8_t mul(uint8_t a, uint8_t b) const { return mul_table[a][b]; }
    inline uint8_t frobenius(uint8_t a) const { return mul_table[a][a]; }
    uint8_t full_trace(uint8_t x) const {
        uint8_t res = x;
        for(int i=1; i<8; ++i) { x = frobenius(x); res ^= x; }
        return res & 1;
    }

    std::vector<double> algebraic_signature(uint8_t x) const {
        return {
            (double)full_trace(x),
            (double)orbits[element_to_orbit_id[x]].degree / 8.0,
            (double)x / 255.0,
            (double)element_to_orbit_id[x] / (double)orbits.size()
        };
    }
};

static GaloisField8 GF8;

// --- Gated RNN with Corrected Online Learning ---
class GaloisRNN {
public:
    int input_size, hidden_size, num_orbits;
    std::vector<double> w_ih, w_hh, w_ho;
    std::vector<double> b_h, b_o;
    std::vector<double> h, last_h;

    GaloisRNN(int in, int hid, int orbits_count)
        : input_size(in), hidden_size(hid), num_orbits(orbits_count) {
        w_ih.resize(in * hid); w_hh.resize(hid * hid); w_ho.resize(hid * num_orbits);
        b_h.assign(hid, 0.0); b_o.assign(num_orbits, 0.0);
        std::default_random_engine gen(1337);
        std::uniform_real_distribution<double> dist(-1.0/std::sqrt(hid), 1.0/std::sqrt(hid));
        for(auto& w : w_ih) w = dist(gen);
        for(auto& w : w_hh) w = dist(gen);
        for(auto& w : w_ho) w = dist(gen);
        h.assign(hid, 0.0);
    }

    std::vector<double> forward(const std::vector<double>& x) {
        last_h = h;
        std::vector<double> next_h(hidden_size, 0.0);
        for(int i=0; i<hidden_size; ++i) {
            double sum = b_h[i];
            for(int j=0; j<input_size; ++j) sum += x[j] * w_ih[j * hidden_size + i];
            for(int j=0; j<hidden_size; ++j) sum += last_h[j] * w_hh[j * hidden_size + i];
            next_h[i] = std::tanh(sum);
        }
        h = next_h;

        std::vector<double> logits(num_orbits, 0.0);
        for(int i=0; i<num_orbits; ++i) {
            double sum = b_o[i];
            for(int j=0; j<hidden_size; ++j) sum += h[j] * w_ho[j * num_orbits + i];
            logits[i] = sum;
        }
        double max_l = *std::max_element(logits.begin(), logits.end());
        double sum_exp = 0;
        for(auto& l : logits) { l = std::exp(l - max_l); sum_exp += l; }
        for(auto& l : logits) l /= sum_exp;
        return logits;
    }

    void train(const std::vector<double>& x, int target_orbit, double lr) {
        // Corrected: use cached hidden state from forward pass
        // No redundant forward call to avoid double step
        std::vector<double> probs(num_orbits, 0.0);
        for(int i=0; i<num_orbits; ++i) {
            double sum = b_o[i];
            for(int j=0; j<hidden_size; ++j) sum += h[j] * w_ho[j * num_orbits + i];
            probs[i] = sum;
        }
        double max_l = *std::max_element(probs.begin(), probs.end());
        double sum_exp = 0;
        for(auto& l : probs) { l = std::exp(l - max_l); sum_exp += l; }
        for(auto& l : probs) l /= sum_exp;

        std::vector<double> d_logits = probs;
        d_logits[target_orbit] -= 1.0;

        std::vector<double> d_h(hidden_size, 0.0);
        // Corrected: capture weights before update for backprop
        std::vector<double> old_w_ho = w_ho;

        for(int i=0; i<num_orbits; ++i) {
            b_o[i] -= lr * d_logits[i];
            for(int j=0; j<hidden_size; ++j) {
                w_ho[j * num_orbits + i] -= lr * d_logits[i] * h[j];
                d_h[j] += d_logits[i] * old_w_ho[j * num_orbits + i];
            }
        }

        for(int i=0; i<hidden_size; ++i) {
            double d_pre = d_h[i] * (1.0 - h[i] * h[i]);
            b_h[i] -= lr * d_pre;
            for(int j=0; j<input_size; ++j) w_ih[j * hidden_size + i] -= lr * d_pre * x[j];
            for(int j=0; j<hidden_size; ++j) w_hh[j * hidden_size + i] -= lr * d_pre * last_h[j];
        }
    }
};

// --- LZMA Utils ---
inline std::vector<uint8_t> lzma_compress(const std::vector<uint8_t>& data) {
    lzma_stream strm = LZMA_STREAM_INIT;
    if (lzma_easy_encoder(&strm, 9, LZMA_CHECK_CRC64) != LZMA_OK) return {};
    std::vector<uint8_t> out(data.size() + 1024);
    strm.next_in = data.data(); strm.avail_in = data.size();
    strm.next_out = out.data(); strm.avail_out = out.size();
    if (lzma_code(&strm, LZMA_FINISH) != LZMA_STREAM_END) { lzma_end(&strm); return {}; }
    out.resize(out.size() - strm.avail_out);
    lzma_end(&strm);
    return out;
}

#endif
