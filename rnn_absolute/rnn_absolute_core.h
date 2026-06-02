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
    uint8_t to_normal[256];
    uint8_t from_normal[256];

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
            // Removed std::sort to preserve Frobenius-cyclic order
            for (size_t j = 0; j < orb.elements.size(); j++) {
                element_to_orbit_id[orb.elements[j]] = orb.id;
                element_to_root_index[orb.elements[j]] = (int)j;
            }
            orbits.push_back(orb);
        }

        // Setup Normal Basis transformation (using element 0x20 as normal element for AES poly)
        uint8_t alpha = 0x20;
        uint8_t basis[8];
        basis[0] = alpha;
        for(int i=1; i<8; ++i) basis[i] = frobenius(basis[i-1]);

        for(int i=0; i<256; ++i) {
            uint8_t res = 0;
            for(int j=0; j<8; ++j) {
                if(bilinear_trace((uint8_t)i, basis[j])) res |= (1 << j);
            }
            to_normal[i] = res;
            from_normal[res] = (uint8_t)i;
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
    uint8_t pow(uint8_t a, uint8_t p) const {
        if (p == 0) return 1;
        if (a == 0) return 0;
        uint32_t l = (uint32_t)log_table[a] * p;
        return exp_table[l % 255];
    }
    uint8_t tr8_power(uint8_t x, uint8_t p) const {
        uint8_t v = pow(x, p);
        uint8_t res = v; for(int i=1; i<8; ++i) { v = frobenius(v); res ^= v; }
        return res & 1;
    }
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

inline double estimate_local_fd(const std::vector<uint8_t>& img, int x, int y, int w, int h) {
    auto get = [&](int tx, int ty) { return (tx<0||ty<0||tx>=w||ty>=h)?0:img[ty*w+tx]; };
    auto var = [&](int sz) {
        double s=0, s2=0; int n=0;
        for(int j=y-sz; j<=y+sz; ++j) for(int i=x-sz; i<=x+sz; ++i) {
            double v = (double)get(i, j); s += v; s2 += v*v; n++;
        }
        return (s2/n - (s/n)*(s/n));
    };
    double v1 = var(1), v4 = var(4);
    if(v1 < 1e-5) return 2.0;
    double h_exp = (std::log(v4+1e-5) - std::log(v1+1e-5)) / (2.0 * std::log(4.0));
    return std::clamp(3.0 - h_exp, 1.0, 3.0);
}

inline double estimate_algebraic_complexity(const std::vector<uint8_t>& img, int x, int y, int w, int h) {
    auto get = [&](int tx, int ty) { if(tx<0||ty<0||tx>=w||ty>=h) return (uint8_t)0; return img[ty*w+tx]; };
    auto get_deg = [&](int tx, int ty) {
        uint8_t v = get(tx, ty);
        return (double)GF8.orbits[GF8.element_to_orbit_id[v]].elements.size();
    };
    double d0 = get_deg(x, y);
    double d1 = (get_deg(x-1, y) + get_deg(x, y-1) + get_deg(x+1, y) + get_deg(x, y+1)) / 4.0;
    double d2 = (get_deg(x-2, y) + get_deg(x, y-2) + get_deg(x+2, y) + get_deg(x, y+2)) / 4.0;
    return std::clamp((std::abs(d1 - d0) + std::abs(d2 - d1)) / 8.0, 0.0, 1.0);
}

struct SignalContext {
    int w, h;
    const std::vector<uint8_t>& data;

    uint8_t getV(int tx, int ty) const { return (tx<0||ty<0||tx>=w||ty>=h)?0:data[ty * w + tx]; }

    std::vector<double> stalk(uint8_t v, uint8_t ref, int tx, int ty) const {
        uint8_t a = getV(tx-1, ty), b = getV(tx, ty-1), c = getV(tx-1, ty-1);
        double med = (c >= std::max(a, b)) ? std::min(a, b) : ((c <= std::min(a, b)) ? std::max(a, b) : (a + b - c));
        double gh = (double)(a - c), gv = (double)(b - c);
        uint8_t tr1 = GF8.tr8_1(v), tr2 = GF8.tr8_2(v), tr4 = GF8.tr8_4(v);
        return std::vector<double>{
            (double)tr1, (double)tr4/255.0, (double)tr2/255.0, (double)GF8.norm8_4(v)/255.0,
            (double)GF8.element_to_orbit_id[v]/(double)GF8.orbits.size(),
            (double)GF8.orbits[GF8.element_to_orbit_id[v]].elements.size()/8.0,
            (double)v/255.0, (double)GF8.bilinear_trace(v, ref), (double)GF8.mul(v, GF8.inv_table[ref])/255.0,
            (double)GF8.tr8_power(v, 3), (double)GF8.tr8_power(v, 5), (double)(tr1 ^ (tr2 & 1)), (double)(tr2 ^ tr4)/255.0,
            (double)(tr4 ^ v)/255.0, (double)GF8.to_normal[v]/255.0, med/255.0, gh/255.0, gv/255.0
        };
    }

    void get_features(int x, int y, uint8_t last_rank, uint8_t last_rid, std::vector<double>& x_s, std::vector<double>& x_f) const {
        int nx[] = {x-1, x, x-1, x+1, x-2, x, x-2, x-1}, ny[] = {y, y-1, y-1, y-1, y, y-2, y-1, y-2};
        uint8_t ref = getV(x-1, y-1);
        for(int j=0; j<8; ++j) { auto s = stalk(getV(nx[j], ny[j]), ref, nx[j], ny[j]); x_s.insert(x_s.end(), s.begin(), s.end()); }
        x_s.push_back((double)last_rank/255.0); x_s.push_back((double)last_rid/8.0);
        int fx[] = {x-2, x, x-4, x, x-16}, fy[] = {y, y-2, y, y-4, y};
        for(int j=0; j<5; ++j) {
            uint8_t fv = getV(fx[j], fy[j]); auto s = stalk(fv, ref, fx[j], fy[j]);
            x_f.insert(x_f.end(), s.begin(), s.begin() + 15);
            x_f.push_back(estimate_local_fd(data, fx[j], fy[j], w, h)/3.0);
            x_f.push_back(estimate_algebraic_complexity(data, fx[j], fy[j], w, h));
            x_f.push_back((double)GF8.tr8_power(fv, 7));
        }
        x_f.push_back((double)x/w); x_f.push_back((double)y/h);
        const auto& mp_prev = GF8.orbits[GF8.element_to_orbit_id[getV(x-1, y)]].min_poly_coeffs;
        const auto& mp_curr = GF8.orbits[GF8.element_to_orbit_id[getV(x, y-1)]].min_poly_coeffs;
        for(int k=0; k<8; k++) {
            uint8_t v1 = (k < (int)mp_prev.size()) ? mp_prev[k] : 0;
            uint8_t v2 = (k < (int)mp_curr.size()) ? mp_curr[k] : 0;
            x_f.push_back((double)(v1 ^ v2) / 255.0);
        }
    }
};

// --- Multi-Head Gated RNN with Complexity Prediction ---
class DualPathGaloisGRU {
public:
    struct Prediction { std::vector<double> p_orb, p_root; double complexity; };

    int in_spatial, in_fractal, hidden_size, num_orbits;
    std::vector<double> w_iz, w_ir, w_in, w_hz, w_hr, w_hn;
    std::vector<double> w_gate_s, w_gate_f; // Gated fusion weights
    std::vector<double> b_z, b_r, b_n, b_gate;
    std::vector<double> w_ho_orb, w_ho_root, w_ho_comp;
    std::vector<double> b_o_orb, b_o_root, b_o_comp;
    // RMSprop cache
    std::vector<double> g_w_iz, g_w_ir, g_w_in, g_w_hz, g_w_hr, g_w_hn;
    std::vector<double> g_w_gate_s, g_w_gate_f;
    std::vector<double> g_b_z, g_b_r, g_b_n, g_b_gate;
    std::vector<double> g_w_ho_orb, g_w_ho_root, g_w_ho_comp;
    std::vector<double> g_b_o_orb, g_b_o_root, g_b_o_comp;
    std::vector<double> h, last_h, last_x;
    std::vector<double> cached_z, cached_r, cached_n;
    Prediction last_pred;

    DualPathGaloisGRU(int in_s, int in_f, int hid, int orbits_count)
        : in_spatial(in_s), in_fractal(in_f), hidden_size(hid), num_orbits(orbits_count) {
        int in_total = hid; // Input to GRU is the fused vector
        w_iz.resize(in_total * hid); w_ir.resize(in_total * hid); w_in.resize(in_total * hid);
        w_hz.resize(hid * hid); w_hr.resize(hid * hid); w_hn.resize(hid * hid);
        b_z.assign(hid, 0.0); b_r.assign(hid, 0.0); b_n.assign(hid, 0.0);
        w_ho_orb.resize(hid * num_orbits); w_ho_root.resize(hid * 8); w_ho_comp.resize(hid);
        b_o_orb.assign(num_orbits, 0.0); b_o_root.assign(8, 0.0); b_o_comp.assign(1, 0.0);
        w_gate_s.resize(in_s * hid); w_gate_f.resize(in_f * hid); b_gate.assign(hid, 0.0);

        g_w_iz.assign(in_total * hid, 0.0); g_w_ir.assign(in_total * hid, 0.0); g_w_in.assign(in_total * hid, 0.0);
        g_w_hz.assign(hid * hid, 0.0); g_w_hr.assign(hid * hid, 0.0); g_w_hn.assign(hid * hid, 0.0);
        g_b_z.assign(hid, 0.0); g_b_r.assign(hid, 0.0); g_b_n.assign(hid, 0.0);
        g_w_gate_s.assign(in_s * hid, 0.0); g_w_gate_f.assign(in_f * hid, 0.0); g_b_gate.assign(hid, 0.0);
        g_w_ho_orb.assign(hid * num_orbits, 0.0); g_w_ho_root.assign(hid * 8, 0.0); g_w_ho_comp.assign(hid, 0.0);
        g_b_o_orb.assign(num_orbits, 0.0); g_b_o_root.assign(8, 0.0); g_b_o_comp.assign(1, 0.0);

        std::default_random_engine gen(1337);
        std::uniform_real_distribution<double> dist(-0.1, 0.1);
        std::normal_distribution<double> norm(0.0, 1.0);
        auto init = [&](std::vector<double>& v) { for(auto& x : v) x = dist(gen); };
        init(w_iz); init(w_ir); init(w_in);
        init(w_ho_orb); init(w_ho_root); init(w_ho_comp);
        init(w_gate_s); init(w_gate_f);

        auto ortho_init = [&](std::vector<double>& v, int rows, int cols) {
            for(auto& x : v) x = norm(gen);
            // Simple Gram-Schmidt for orthogonalization
            for(int i=0; i<cols; ++i) {
                for(int j=0; j<i; ++j) {
                    double dot = 0;
                    for(int k=0; k<rows; ++k) dot += v[k*cols+i] * v[k*cols+j];
                    for(int k=0; k<rows; ++k) v[k*cols+i] -= dot * v[k*cols+j];
                }
                double norm_v = 0;
                for(int k=0; k<rows; ++k) norm_v += v[k*cols+i] * v[k*cols+i];
                norm_v = std::sqrt(norm_v + 1e-9);
                for(int k=0; k<rows; ++k) v[k*cols+i] /= norm_v;
            }
        };
        ortho_init(w_hz, hidden_size, hidden_size);
        ortho_init(w_hr, hidden_size, hidden_size);
        ortho_init(w_hn, hidden_size, hidden_size);
        h.assign(hid, 0.0);
        cached_z.assign(hid, 0.0); cached_r.assign(hid, 0.0); cached_n.assign(hid, 0.0);
    }

    Prediction forward(const std::vector<double>& x_s, const std::vector<double>& x_f) {
        last_h = h;
        auto sigmoid = [](double v) { return 1.0 / (1.0 + std::exp(-std::clamp(v, -15.0, 15.0))); };

        // Optimized Gated Fusion of Spatial and Fractal paths
        std::vector<double> x_fused(hidden_size);
        for(int i=0; i<hidden_size; ++i) {
            double s_val = 0, f_val = 0;
            for(int j=0; j<in_spatial; ++j) s_val += x_s[j] * w_gate_s[j*hidden_size+i];
            for(int j=0; j<in_fractal; ++j) f_val += x_f[j] * w_gate_f[j*hidden_size+i];
            double g = b_gate[i] + s_val + f_val;
            double gate_val = sigmoid(g);
            x_fused[i] = gate_val * s_val + (1.0 - gate_val) * f_val;
        }

        last_x = x_fused;
        int in_total = hidden_size;

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

    void train(const std::vector<double>& x_s, const std::vector<double>& x_f, int t_orb, int t_root, double t_comp, double lr) {
        double eps = 0.1, rho = 0.9, e = 1e-8;
        auto step = [&](double& w, double& g_cache, double grad) {
            grad = std::clamp(grad, -1.0, 1.0); // Simple gradient clipping
            g_cache = rho * g_cache + (1.0 - rho) * grad * grad;
            w -= (lr * grad / (std::sqrt(g_cache) + e)) + 1e-4 * w; // L2 decay
        };

        std::vector<double> d_orb(num_orbits);
        for(int i=0; i<num_orbits; ++i) d_orb[i] = last_pred.p_orb[i] - ((i == t_orb) ? (1.0 - eps) : (eps / (num_orbits - 1)));
        std::vector<double> d_root(8);
        for(int i=0; i<8; ++i) d_root[i] = last_pred.p_root[i] - ((i == t_root) ? (1.0 - eps) : (eps / 7.0));
        double d_comp_val = last_pred.complexity - t_comp;

        std::vector<double> d_h(hidden_size, 0.0);
        for(int i=0; i<num_orbits; ++i) {
            step(b_o_orb[i], g_b_o_orb[i], d_orb[i]);
            for(int j=0; j<hidden_size; ++j) {
                d_h[j] += d_orb[i] * w_ho_orb[j*num_orbits+i];
                step(w_ho_orb[j*num_orbits+i], g_w_ho_orb[j*num_orbits+i], d_orb[i] * h[j]);
            }
        }
        for(int i=0; i<8; ++i) {
            step(b_o_root[i], g_b_o_root[i], d_root[i]);
            for(int j=0; j<hidden_size; ++j) {
                d_h[j] += d_root[i] * w_ho_root[j*8+i];
                step(w_ho_root[j*8+i], g_w_ho_root[j*8+i], d_root[i] * h[j]);
            }
        }
        step(b_o_comp[0], g_b_o_comp[0], d_comp_val);
        double d_comp_tanh = d_comp_val * (1.0 - last_pred.complexity * last_pred.complexity);
        for(int j=0; j<hidden_size; ++j) {
            d_h[j] += d_comp_tanh * w_ho_comp[j];
            step(w_ho_comp[j], g_w_ho_comp[j], d_comp_tanh * h[j]);
        }

        std::vector<double> d_z(hidden_size), d_r(hidden_size), d_n(hidden_size);
        for(int i=0; i<hidden_size; ++i) {
            d_z[i] = d_h[i] * (cached_n[i] - last_h[i]) * (cached_z[i] * (1.0 - cached_z[i]));
            d_n[i] = d_h[i] * cached_z[i] * (1.0 - cached_n[i] * cached_n[i]);
        }

        std::vector<double> d_x_fused(hidden_size, 0.0);
        int in_total = hidden_size;
        for(int i=0; i<hidden_size; ++i) {
            step(b_z[i], g_b_z[i], d_z[i]);
            step(b_n[i], g_b_n[i], d_n[i]);
            for(int j=0; j<in_total; ++j) {
                d_x_fused[j] += d_z[i] * w_iz[j*hidden_size+i] + d_n[i] * w_in[j*hidden_size+i];
                step(w_iz[j*hidden_size+i], g_w_iz[j*hidden_size+i], d_z[i] * last_x[j]);
                step(w_in[j*hidden_size+i], g_w_in[j*hidden_size+i], d_n[i] * last_x[j]);
            }
            for(int j=0; j<hidden_size; ++j) {
                step(w_hz[j*hidden_size+i], g_w_hz[j*hidden_size+i], d_z[i] * last_h[j]);
                step(w_hn[j*hidden_size+i], g_w_hn[j*hidden_size+i], d_n[i] * (cached_r[j] * last_h[j]));
            }
        }

        for(int i=0; i<hidden_size; ++i) {
            double d_rec_n = 0;
            for(int j=0; j<hidden_size; ++j) d_rec_n += d_n[j] * w_hn[i*hidden_size+j];
            d_r[i] = d_rec_n * last_h[i] * (cached_r[i] * (1.0 - cached_r[i]));
        }

        for(int i=0; i<hidden_size; ++i) {
            step(b_r[i], g_b_r[i], d_r[i]);
            for(int j=0; j<in_total; ++j) {
                d_x_fused[j] += d_r[i] * w_ir[j*hidden_size+i];
                step(w_ir[j*hidden_size+i], g_w_ir[j*hidden_size+i], d_r[i] * last_x[j]);
            }
            for(int j=0; j<hidden_size; ++j) step(w_hr[j*hidden_size+i], g_w_hr[j*hidden_size+i], d_r[i] * last_h[j]);
        }

        // Backprop through Gated Fusion
        auto sigmoid = [](double v) { return 1.0 / (1.0 + std::exp(-std::clamp(v, -15.0, 15.0))); };
        for(int i=0; i<hidden_size; ++i) {
            double sz = 0, fz = 0;
            for(int j=0; j<in_spatial; ++j) sz += x_s[j] * w_gate_s[j*hidden_size+i];
            for(int j=0; j<in_fractal; ++j) fz += x_f[j] * w_gate_f[j*hidden_size+i];
            double g = b_gate[i] + sz + fz;
            double sig = sigmoid(g);
            double d_g = d_x_fused[i] * (sz - fz) * sig * (1.0 - sig);
            step(b_gate[i], g_b_gate[i], d_g);
            for(int j=0; j<in_spatial; ++j) {
                double grad_w = d_g * x_s[j] + d_x_fused[i] * sig * x_s[j];
                step(w_gate_s[j*hidden_size+i], g_w_gate_s[j*hidden_size+i], grad_w);
            }
            for(int j=0; j<in_fractal; ++j) {
                double grad_w = d_g * x_f[j] + d_x_fused[i] * (1.0 - sig) * x_f[j];
                step(w_gate_f[j*hidden_size+i], g_w_gate_f[j*hidden_size+i], grad_w);
            }
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

inline std::vector<std::vector<uint8_t>> lzma_decompress_channels(const std::vector<uint8_t>& compressed) {
    lzma_stream strm = LZMA_STREAM_INIT;
    if (lzma_stream_decoder(&strm, UINT64_MAX, 0) != LZMA_OK) return {};
    std::vector<uint8_t> all_data;
    std::vector<uint8_t> buf(65536);
    strm.next_in = compressed.data(); strm.avail_in = compressed.size();
    lzma_ret ret;
    do {
        strm.next_out = buf.data(); strm.avail_out = buf.size();
        ret = lzma_code(&strm, LZMA_RUN);
        all_data.insert(all_data.end(), buf.begin(), buf.begin() + (buf.size() - strm.avail_out));
    } while (ret == LZMA_OK);
    lzma_end(&strm);
    if (ret != LZMA_STREAM_END) return {};

    std::vector<std::vector<uint8_t>> channels;
    size_t pos = 0;
    while(pos + 4 <= all_data.size()) {
        uint32_t sz = all_data[pos] | (all_data[pos+1] << 8) | (all_data[pos+2] << 16) | (all_data[pos+3] << 24);
        pos += 4;
        if (pos + sz > all_data.size()) break;
        channels.push_back(std::vector<uint8_t>(all_data.begin() + pos, all_data.begin() + pos + sz));
        pos += sz;
    }
    return channels;
}

#endif
