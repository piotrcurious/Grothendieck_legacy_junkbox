#ifndef RNN_ABSOLUTE_CORE_H
#define RNN_ABSOLUTE_CORE_H

#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <cstdint>
#include <lzma.h>

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

// --- Finite Field GF(2^8) Logic ---
const uint16_t AES_POLY = 0x11B;

inline uint8_t gf_mul(uint8_t a, uint8_t b) {
    uint8_t res = 0;
    for (int i = 0; i < 8; ++i) {
        if (b & 1) res ^= a;
        bool hi = (a & 0x80);
        a <<= 1;
        if (hi) a ^= (AES_POLY & 0xFF);
        b >>= 1;
    }
    return res;
}

inline uint8_t gf_pow2(uint8_t a) { return gf_mul(a, a); }

inline uint8_t frobenius(uint8_t a, int k) {
    for (int i = 0; i < k; ++i) a = gf_pow2(a);
    return a;
}

// --- Galois Canonicalization ---
struct GaloisOrbit {
    std::vector<uint8_t> canonical;
    int k; // Frobenius power used to reach canonical
};

inline GaloisOrbit get_canonical(const std::vector<uint8_t>& vec) {
    std::vector<uint8_t> best = vec;
    int best_k = 0;
    std::vector<uint8_t> cur = vec;
    for (int k = 1; k < 8; ++k) {
        for (auto& x : cur) x = gf_pow2(x);
        if (cur < best) {
            best = cur;
            best_k = k;
        }
    }
    return {best, best_k};
}

// --- Hilbert/Morton Utilities ---
inline uint32_t expandBits(uint32_t v) {
    v = (v | (v << 8)) & 0x00FF00FFu;
    v = (v | (v << 4)) & 0x0F0F0F0Fu;
    v = (v | (v << 2)) & 0x33333333u;
    v = (v | (v << 1)) & 0x55555555u;
    return v;
}

inline uint32_t morton2D(uint32_t x, uint32_t y) {
    return (expandBits(y) << 1) | expandBits(x);
}

inline void rot(int n, int *x, int *y, int rx, int ry) {
    if (ry == 0) {
        if (rx == 1) {
            *x = n-1 - *x;
            *y = n-1 - *y;
        }
        int t  = *x;
        *x = *y;
        *y = t;
    }
}

inline int64_t hilbert2D(int n, int x, int y) {
    int rx, ry;
    int64_t d=0;
    for (int s=n/2; s>0; s/=2) {
        rx = (x & s) > 0;
        ry = (y & s) > 0;
        d += (int64_t)s * s * ((3 * rx) ^ ry);
        rot(s, &x, &y, rx, ry);
    }
    return d;
}

// --- GatedRNN with robust online single-step gradients ---
class GatedRNN {
public:
    GatedRNN(int in, int hid) : input_size(in), hidden_size(hid) {
        int w_size = (in + hid) * hid;
        w_z.assign(w_size, 0.0); w_r.assign(w_size, 0.0); w_h.assign(w_size, 0.0);
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
        last_x = x;
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
            for(int i=0; i<hidden_size; ++i) {
                dr += d_pre_h[i] * w_h[(input_size + j) * hidden_size + i] * last_h[j];
            }
            double d_pre_r_j = dr * r[j] * (1.0 - r[j]);
            b_r[j] -= lr * d_pre_r_j;
            for(int k=0; k<input_size; ++k) w_r[k * hidden_size + j] -= lr * d_pre_r_j * x[k];
            for(int k=0; k<hidden_size; ++k) w_r[(input_size + k) * hidden_size + j] -= lr * d_pre_r_j * last_h[k];
        }
    }

private:
    int input_size, hidden_size;
    std::vector<double> w_z, w_r, w_h, w_o;
    std::vector<double> b_z, b_r, b_h;
    double b_o;
    std::vector<double> h, last_h, last_x;
    std::vector<double> z, r, h_tilde;
};

// --- LZMA Compression Wrap ---
inline std::vector<uint8_t> lzma_compress(const std::vector<uint8_t>& data) {
    lzma_stream strm = LZMA_STREAM_INIT;
    if (lzma_easy_encoder(&strm, 9, LZMA_CHECK_CRC64) != LZMA_OK) return {};

    std::vector<uint8_t> out(data.size() + 1024);
    strm.next_in = data.data();
    strm.avail_in = data.size();
    strm.next_out = out.data();
    strm.avail_out = out.size();

    if (lzma_code(&strm, LZMA_FINISH) != LZMA_STREAM_END) {
        lzma_end(&strm);
        return {};
    }
    out.resize(out.size() - strm.avail_out);
    lzma_end(&strm);
    return out;
}

#endif
