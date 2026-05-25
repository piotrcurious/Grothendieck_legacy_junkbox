#ifndef RNN_ABSOLUTE_CORE_H
#define RNN_ABSOLUTE_CORE_H

#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <cstdint>
#include <lzma.h>
#include <map>

// --- Comprehensive Galois Field GF(2^8) with Absolute Group Action ---
class GaloisField8 {
public:
    static const uint16_t POLY = 0x11B;
    uint8_t mul_table[256][256];
    uint8_t exp_table[256];
    uint8_t log_table[256];
    uint8_t inv_table[256];

    GaloisField8() {
        uint16_t x = 1;
        for (int i = 0; i < 255; i++) {
            exp_table[i] = (uint8_t)x;
            log_table[x] = (uint8_t)i;
            x <<= 1;
            if (x & 0x100) x ^= POLY;
        }
        exp_table[255] = exp_table[0];
        log_table[0] = 0;

        for (int i = 0; i < 256; i++) {
            for (int j = 0; j < 256; j++) {
                mul_table[i][j] = compute_mul(i, j);
            }
        }
        inv_table[0] = 0;
        for (int i = 1; i < 256; i++) {
            for (int j = 1; j < 256; j++) {
                if (mul_table[i][j] == 1) {
                    inv_table[i] = (uint8_t)j;
                    break;
                }
            }
        }
    }

    uint8_t compute_mul(uint8_t a, uint8_t b) {
        uint8_t res = 0;
        for (int i = 0; i < 8; i++) {
            if (b & 1) res ^= a;
            bool hi = (a & 0x80);
            a <<= 1;
            if (hi) a ^= (POLY & 0xFF);
            b >>= 1;
        }
        return res;
    }

    inline uint8_t mul(uint8_t a, uint8_t b) const { return mul_table[a][b]; }
    inline uint8_t add(uint8_t a, uint8_t b) const { return a ^ b; }
    inline uint8_t div(uint8_t a, uint8_t b) const { return mul_table[a][inv_table[b]]; }
    inline uint8_t frobenius(uint8_t a) const { return mul_table[a][a]; }

    // Trace from GF(2^8) to GF(2)
    // Tr(a) = a + a^2 + a^4 + a^8 + a^16 + a^32 + a^64 + a^128
    uint8_t trace(uint8_t a) const {
        uint8_t res = a;
        uint8_t cur = a;
        for(int i=0; i<7; ++i) {
            cur = frobenius(cur);
            res ^= cur;
        }
        return res & 1;
    }

    // Canonical representative under Absolute Galois Group action (Frobenius orbits)
    uint8_t get_canonical(uint8_t a, int& k_out) const {
        uint8_t best = a;
        k_out = 0;
        uint8_t cur = a;
        for (int k = 1; k < 8; k++) {
            cur = frobenius(cur);
            if (cur < best) {
                best = cur;
                k_out = k;
            }
        }
        return best;
    }
};

static GaloisField8 GF8;

// --- Galois-Equivariant GatedRNN ---
class GaloisGatedRNN {
public:
    int input_size, hidden_size;
    std::vector<double> w_z, w_r, w_h, w_o;
    std::vector<double> b_z, b_r, b_h;
    double b_o;
    std::vector<double> h, last_h;
    std::vector<double> z, r, h_tilde;

    GaloisGatedRNN(int in, int hid) : input_size(in), hidden_size(hid) {
        int w_size = (in + hid) * hid;
        w_z.resize(w_size); w_r.resize(w_size); w_h.resize(w_size);
        b_z.assign(hid, 0.0); b_r.assign(hid, 0.0); b_h.assign(hid, 0.0);
        w_o.assign(hid, 0.0); b_o = 0.0;

        std::default_random_engine gen(1337);
        std::uniform_real_distribution<double> dist(-0.1, 0.1);
        for(auto& w : w_z) w = dist(gen);
        for(auto& w : w_r) w = dist(gen);
        for(auto& w : w_h) w = dist(gen);
        for(auto& w : w_o) w = dist(gen);

        h.assign(hid, 0.0);
        last_h.assign(hid, 0.0);
    }

    double forward(const std::vector<double>& x) {
        last_h = h;
        z.assign(hidden_size, 0.0); r.assign(hidden_size, 0.0); h_tilde.assign(hidden_size, 0.0);
        auto sigmoid = [](double v) { return 1.0 / (1.0 + std::exp(-v)); };

        for(int i=0; i<hidden_size; ++i) {
            double sum_z = b_z[i], sum_r = b_r[i];
            for(int j=0; j<input_size; ++j) {
                sum_z += x[j] * w_z[j * hidden_size + i];
                sum_r += x[j] * w_r[j * hidden_size + i];
            }
            for(int j=0; j<hidden_size; ++j) {
                sum_z += last_h[j] * w_z[(input_size + j) * hidden_size + i];
                sum_r += last_h[j] * w_r[(input_size + j) * hidden_size + i];
            }
            z[i] = sigmoid(sum_z);
            r[i] = sigmoid(sum_r);
        }

        for(int i=0; i<hidden_size; ++i) {
            double sum_h = b_h[i];
            for(int j=0; j<input_size; ++j) sum_h += x[j] * w_h[j * hidden_size + i];
            for(int j=0; j<hidden_size; ++j) sum_h += (r[j] * last_h[j]) * w_h[(input_size + j) * hidden_size + i];
            h_tilde[i] = std::tanh(sum_h);
        }

        for(int i=0; i<hidden_size; ++i) {
            h[i] = (1.0 - z[i]) * last_h[i] + z[i] * h_tilde[i];
        }

        double out = b_o;
        for(int i=0; i<hidden_size; ++i) out += h[i] * w_o[i];
        return out;
    }

    void train(const std::vector<double>& x, double target, double lr) {
        double out = b_o;
        for(int i=0; i<hidden_size; ++i) out += h[i] * w_o[i];
        double dy = out - target;

        b_o -= lr * dy;
        std::vector<double> d_h_global(hidden_size);
        for(int i=0; i<hidden_size; ++i) {
            d_h_global[i] = dy * w_o[i];
            w_o[i] -= lr * dy * h[i];
        }

        std::vector<double> d_pre_h(hidden_size);
        std::vector<double> d_pre_z(hidden_size);

        for(int i=0; i<hidden_size; ++i) {
            d_pre_h[i] = d_h_global[i] * z[i] * (1.0 - h_tilde[i] * h_tilde[i]);
            d_pre_z[i] = d_h_global[i] * (h_tilde[i] - last_h[i]) * z[i] * (1.0 - z[i]);
        }

        for(int i=0; i<hidden_size; ++i) {
            b_h[i] -= lr * d_pre_h[i];
            b_z[i] -= lr * d_pre_z[i];
            for(int j=0; j<input_size; ++j) {
                w_h[j * hidden_size + i] -= lr * d_pre_h[i] * x[j];
                w_z[j * hidden_size + i] -= lr * d_pre_z[i] * x[j];
            }
            for(int j=0; j<hidden_size; ++j) {
                w_h[(input_size + j) * hidden_size + i] -= lr * d_pre_h[i] * (r[j] * last_h[j]);
                w_z[(input_size + j) * hidden_size + i] -= lr * d_pre_z[i] * last_h[j];
            }
        }

        for(int j=0; j<hidden_size; ++j) {
            double dr = 0;
            for(int i=0; i<hidden_size; ++i) dr += d_pre_h[i] * w_h[(input_size + j) * hidden_size + i] * last_h[j];
            double d_pre_r_j = dr * r[j] * (1.0 - r[j]);
            b_r[j] -= lr * d_pre_r_j;
            for(int k=0; k<input_size; ++k) w_r[k * hidden_size + j] -= lr * d_pre_r_j * x[k];
            for(int k=0; k<hidden_size; ++k) w_r[(input_size + k) * hidden_size + j] -= lr * d_pre_r_j * last_h[k];
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
