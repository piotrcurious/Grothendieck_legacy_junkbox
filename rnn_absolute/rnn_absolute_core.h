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
        std::vector<uint8_t> min_poly_coeffs;
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
            std::vector<uint8_t> poly = {1};
            for(uint8_t e : orb.elements) {
                std::vector<uint8_t> next_poly(poly.size() + 1, 0);
                for(size_t k=0; k<poly.size(); k++) {
                    next_poly[k+1] ^= poly[k];
                    next_poly[k] ^= mul_table[e][poly[k]];
                }
                poly = next_poly;
            }
            orb.min_poly_coeffs = poly;
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
    uint8_t tr8_4(uint8_t x) const {
        uint8_t xq = x; for(int i=0; i<4; ++i) xq = frobenius(xq);
        return x ^ xq;
    }
    uint8_t tr8_2(uint8_t x) const {
        uint8_t res = x;
        uint8_t xq = x; for(int i=0; i<2; ++i) xq = frobenius(xq); res ^= xq;
        for(int i=0; i<2; ++i) xq = frobenius(xq); res ^= xq;
        for(int i=0; i<2; ++i) xq = frobenius(xq); res ^= xq;
        return res;
    }
    uint8_t tr8_1(uint8_t x) const {
        uint8_t res = x; for(int i=1; i<8; ++i) { x = frobenius(x); res ^= x; }
        return res & 1;
    }
    uint8_t norm8_4(uint8_t x) const {
        uint8_t xq = x; for(int i=0; i<4; ++i) xq = frobenius(xq);
        return mul(x, xq);
    }
    uint8_t bilinear_trace(uint8_t a, uint8_t b) const { return tr8_1(mul(a, b)); }
};

static GaloisField8 GF8;

// --- Multi-Head Gated RNN with Complexity Prediction ---
class DualPathGaloisGRU {
public:
    struct Prediction { std::vector<double> p_orb, p_root; double complexity; };

    int in_spatial, in_fractal, hidden_size, num_orbits;
    std::vector<double> w_iz, w_ir, w_in, w_hz, w_hr, w_hn;
    std::vector<double> b_z, b_r, b_n;
    std::vector<double> w_ho_orb, w_ho_root, w_ho_comp;
    std::vector<double> b_o_orb, b_o_root, b_o_comp;
    std::vector<double> h, last_h, last_x;
    std::vector<double> cached_z, cached_r, cached_n;
    Prediction last_pred;

    DualPathGaloisGRU(int in_s, int in_f, int hid, int orbits_count)
        : in_spatial(in_s), in_fractal(in_f), hidden_size(hid), num_orbits(orbits_count) {
        int in_total = in_s + in_f;
        w_iz.resize(in_total * hid); w_ir.resize(in_total * hid); w_in.resize(in_total * hid);
        w_hz.resize(hid * hid); w_hr.resize(hid * hid); w_hn.resize(hid * hid);
        b_z.assign(hid, 0.0); b_r.assign(hid, 0.0); b_n.assign(hid, 0.0);
        w_ho_orb.resize(hid * num_orbits); w_ho_root.resize(hid * 8); w_ho_comp.resize(hid);
        b_o_orb.assign(num_orbits, 0.0); b_o_root.assign(8, 0.0); b_o_comp.assign(1, 0.0);

        std::default_random_engine gen(1337);
        std::uniform_real_distribution<double> dist(-0.1, 0.1);
        auto init = [&](std::vector<double>& v) { for(auto& x : v) x = dist(gen); };
        init(w_iz); init(w_ir); init(w_in); init(w_hz); init(w_hr); init(w_hn);
        init(w_ho_orb); init(w_ho_root); init(w_ho_comp);
        h.assign(hid, 0.0);
        cached_z.assign(hid, 0.0); cached_r.assign(hid, 0.0); cached_n.assign(hid, 0.0);
    }

    Prediction forward(const std::vector<double>& x_s, const std::vector<double>& x_f) {
        last_h = h;
        last_x = x_s; last_x.insert(last_x.end(), x_f.begin(), x_f.end());
        int in_total = (int)last_x.size();

        auto sigmoid = [](double v) { return 1.0 / (1.0 + std::exp(-std::clamp(v, -15.0, 15.0))); };

        for(int i=0; i<hidden_size; ++i) {
            double sz = b_z[i], sr = b_r[i];
            for(int j=0; j<in_total; ++j) {
                sz += last_x[j]*w_iz[j*hidden_size+i];
                sr += last_x[j]*w_ir[j*hidden_size+i];
            }
            for(int j=0; j<hidden_size; ++j) {
                sz += last_h[j]*w_hz[j*hidden_size+i];
                sr += last_h[j]*w_hr[j*hidden_size+i];
            }
            cached_z[i] = sigmoid(sz); cached_r[i] = sigmoid(sr);
        }
        for(int i=0; i<hidden_size; ++i) {
            double sn = b_n[i];
            for(int j=0; j<in_total; ++j) sn += last_x[j]*w_in[j*hidden_size+i];
            for(int j=0; j<hidden_size; ++j) sn += (cached_r[j]*last_h[j])*w_hn[j*hidden_size+i];
            cached_n[i] = std::tanh(sn);
        }
        for(int i=0; i<hidden_size; ++i) h[i] = (1.0 - cached_z[i])*last_h[i] + cached_z[i]*cached_n[i];

        auto softmax = [](std::vector<double>& v) {
            double mv = *std::max_element(v.begin(), v.end());
            double se = 0; for(auto& val : v) { val = std::exp(std::clamp(val-mv, -15.0, 15.0)); se += val; }
            for(auto& val : v) val /= (se + 1e-9);
        };

        std::vector<double> lo_orb(num_orbits, 0.0), lo_root(8, 0.0);
        for(int i=0; i<num_orbits; ++i) {
            double s = b_o_orb[i]; for(int j=0; j<hidden_size; ++j) s += h[j]*w_ho_orb[j*num_orbits+i];
            lo_orb[i] = s;
        }
        for(int i=0; i<8; ++i) {
            double s = b_o_root[i]; for(int j=0; j<hidden_size; ++j) s += h[j]*w_ho_root[j*8+i];
            lo_root[i] = s;
        }
        softmax(lo_orb); softmax(lo_root);
        double comp = b_o_comp[0]; for(int j=0; j<hidden_size; ++j) comp += h[j]*w_ho_comp[j];
        last_pred = {lo_orb, lo_root, std::tanh(comp)};
        return last_pred;
    }

    void train(int t_orb, int t_root, double t_comp, double lr) {
        std::vector<double> d_orb = last_pred.p_orb; d_orb[t_orb] -= 1.0;
        std::vector<double> d_root = last_pred.p_root; d_root[t_root] -= 1.0;
        double d_comp_val = last_pred.complexity - t_comp;

        std::vector<double> d_h(hidden_size, 0.0);
        for(int i=0; i<num_orbits; ++i) {
            b_o_orb[i] -= lr * d_orb[i];
            for(int j=0; j<hidden_size; ++j) {
                d_h[j] += d_orb[i] * w_ho_orb[j*num_orbits+i];
                w_ho_orb[j*num_orbits+i] -= lr * d_orb[i] * h[j];
            }
        }
        for(int i=0; i<8; ++i) {
            b_o_root[i] -= lr * d_root[i];
            for(int j=0; j<hidden_size; ++j) {
                d_h[j] += d_root[i] * w_ho_root[j*8+i];
                w_ho_root[j*8+i] -= lr * d_root[i] * h[j];
            }
        }
        b_o_comp[0] -= lr * d_comp_val;
        double d_comp_tanh = d_comp_val * (1.0 - last_pred.complexity * last_pred.complexity);
        for(int j=0; j<hidden_size; ++j) {
            d_h[j] += d_comp_tanh * w_ho_comp[j];
            w_ho_comp[j] -= lr * d_comp_tanh * h[j];
        }

        // BPTT for GRU gates
        std::vector<double> d_z(hidden_size), d_r(hidden_size), d_n(hidden_size);
        for(int i=0; i<hidden_size; ++i) {
            d_z[i] = d_h[i] * (cached_n[i] - last_h[i]) * (cached_z[i] * (1.0 - cached_z[i]));
            d_n[i] = d_h[i] * cached_z[i] * (1.0 - cached_n[i] * cached_n[i]);
        }

        int in_total = (int)last_x.size();
        for(int i=0; i<hidden_size; ++i) {
            b_z[i] -= lr * d_z[i];
            b_n[i] -= lr * d_n[i];
            for(int j=0; j<in_total; ++j) {
                w_iz[j*hidden_size+i] -= lr * d_z[i] * last_x[j];
                w_in[j*hidden_size+i] -= lr * d_n[i] * last_x[j];
            }
            for(int j=0; j<hidden_size; ++j) {
                w_hz[j*hidden_size+i] -= lr * d_z[i] * last_h[j];
                w_hn[j*hidden_size+i] -= lr * d_n[i] * (cached_r[j] * last_h[j]);
            }
        }

        for(int i=0; i<hidden_size; ++i) {
            double d_rec_n = 0;
            for(int j=0; j<hidden_size; ++j) d_rec_n += d_n[j] * w_hn[i*hidden_size+j];
            d_r[i] = d_rec_n * last_h[i] * (cached_r[i] * (1.0 - cached_r[i]));
        }

        for(int i=0; i<hidden_size; ++i) {
            b_r[i] -= lr * d_r[i];
            for(int j=0; j<in_total; ++j) w_ir[j*hidden_size+i] -= lr * d_r[i] * last_x[j];
            for(int j=0; j<hidden_size; ++j) w_hr[j*hidden_size+i] -= lr * d_r[i] * last_h[j];
        }
    }
};

// --- LZMA Multi-Channel Utils ---
inline std::vector<uint8_t> lzma_compress_channels(const std::vector<std::vector<uint8_t>>& channels) {
    std::vector<uint8_t> all_data;
    for(const auto& c : channels) {
        uint32_t sz = (uint32_t)c.size();
        all_data.push_back(sz & 0xFF);
        all_data.push_back((sz >> 8) & 0xFF);
        all_data.push_back((sz >> 16) & 0xFF);
        all_data.push_back((sz >> 24) & 0xFF);
        all_data.insert(all_data.end(), c.begin(), c.end());
    }
    lzma_stream strm = LZMA_STREAM_INIT;
    if (lzma_easy_encoder(&strm, 9, LZMA_CHECK_CRC64) != LZMA_OK) return {};
    std::vector<uint8_t> out(all_data.size() + 4096);
    strm.next_in = all_data.data(); strm.avail_in = all_data.size();
    strm.next_out = out.data(); strm.avail_out = out.size();
    if (lzma_code(&strm, LZMA_FINISH) != LZMA_STREAM_END) { lzma_end(&strm); return {}; }
    out.resize(out.size() - strm.avail_out);
    lzma_end(&strm);
    return out;
}

#endif
