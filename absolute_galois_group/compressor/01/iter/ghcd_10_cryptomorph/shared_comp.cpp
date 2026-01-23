#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <sstream>
#include <cstdint>
#include <cfloat>
#include <cstring>
#include <iomanip>
#include <limits>

#include <Eigen/Dense>

#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/zlib.hpp>

using namespace Eigen;
using namespace std;

static constexpr double PI = 3.14159265358979323846;

// ---------------- Configuration ----------------
const int CS = 8;     // Canonical atom size
const int Q_ANG = 64; // Quantization buckets for Angle
const int Q_SHF = 16; // Quantization buckets for Shift

struct Config {
    int bs = 16;          // base block
    int rs = 8;           // residual block
    int be = 256;         // base dict max atoms
    int re = 512;         // residual dict max atoms
    float bvt = 40;       // base variance threshold
    float rvt = 10;       // residual variance threshold
    float lb = 200;       // Lagrange base
    float lr = 100;       // Lagrange residual
};

enum QTType : uint8_t { QL = 0, QS = 1 };

#pragma pack(push,1)
struct Entry {
    uint16_t id = 0;   // Atom ID
    uint8_t  ang = 0;  // Quantized Rotation Angle
    uint8_t  sx  = 0;  // Quantized Shift X
    uint8_t  sy  = 0;  // Quantized Shift Y
    uint8_t  flg = 0;  // Flags: bit0=Flip, bit1=Negate
    uint8_t  off = 0;  // DC offset (Mean) - Fixed: Now 0..255 direct
    uint8_t  gn  = 0;  // Gain
};
#pragma pack(pop)

// ---------------- Crypto/Math Analysis Utils ----------------
struct CryptoUtils {
    struct Feature {
        Vector2f center; 
        float angle;     
        float energy;    
    };

    static Feature analyze(const MatrixXf& b) {
        Feature f = { Vector2f(0,0), 0, 0 };
        int N = b.rows();
        float sum = 0, sumX = 0, sumY = 0;
        
        MatrixXf absB = b.cwiseAbs();
        for(int r=0; r<N; ++r) {
            for(int c=0; c<N; ++c) {
                float v = absB(r,c);
                sum += v;
                sumX += c * v;
                sumY += r * v;
            }
        }
        
        float cx = (N-1) * 0.5f;
        float cy = (N-1) * 0.5f;

        if (sum > 1e-5f) {
            f.center = Vector2f(sumX/sum - cx, sumY/sum - cy);
        }

        float gxx = 0, gyy = 0, gxy = 0;
        for(int r=1; r<N-1; ++r) {
            for(int c=1; c<N-1; ++c) {
                float dx = b(r,c+1) - b(r,c-1);
                float dy = b(r+1,c) - b(r-1,c);
                gxx += dx*dx;
                gyy += dy*dy;
                gxy += dx*dy;
            }
        }
        
        f.angle = 0.5f * atan2(2.0f * gxy, gxx - gyy + 1e-9f);
        f.energy = b.squaredNorm();
        return f;
    }

    static float solve_delta_angle(float target_ang, float atom_ang) {
        float d = target_ang - atom_ang;
        while (d <= -PI) d += 2*PI;
        while (d > PI)   d -= 2*PI;
        return d;
    }
};

// ---------------- Morphism Engine ----------------
struct Engine {
    static MatrixXf D;
    static int D_N;

    static void ensureD(int N) {
        if (D_N == N) return;
        D_N = N;
        D = MatrixXf(N,N);
        const float s1 = sqrt(1.0f / N);
        const float s2 = sqrt(2.0f / N);
        for(int k=0;k<N;++k) for(int n=0;n<N;++n)
            D(k,n) = (k==0? s1 : s2) * (float)cos(PI * (n + 0.5) * k / N);
    }

    static MatrixXf dct(const MatrixXf& s, bool inv=false) {
        int N = s.rows(); ensureD(N);
        if (inv) return (D.transpose() * s * D).eval();
        else     return (D * s * D.transpose()).eval();
    }

    static MatrixXf warp(const MatrixXf& src, float angle_rad, float sx, float sy, bool flip) {
        int N = src.rows();
        int M = src.cols();
        MatrixXf dst = MatrixXf::Zero(N,M);
        
        float cx = (M-1) * 0.5f;
        float cy = (N-1) * 0.5f;
        float c = cos(angle_rad);
        float s = sin(angle_rad);

        for(int y=0; y<N; ++y) {
            for(int x=0; x<M; ++x) {
                float tx = x - cx;
                float ty = y - cy;
                float ux = tx - sx;
                float uy = ty - sy;
                float rx = ux * c + uy * s;
                float ry = -ux * s + uy * c;
                if (flip) rx = -rx;
                float src_x = rx + cx;
                float src_y = ry + cy;

                int x0 = (int)floor(src_x);
                int y0 = (int)floor(src_y);
                int x1 = x0 + 1;
                int y1 = y0 + 1;

                auto get_pix = [&](int r, int c_idx) -> float {
                    if (r < 0) r = 0; if (r >= N) r = N-1;
                    if (c_idx < 0) c_idx = 0; if (c_idx >= M) c_idx = M-1;
                    return src(r,c_idx);
                };

                float v00 = get_pix(y0, x0);
                float v10 = get_pix(y0, x1);
                float v01 = get_pix(y1, x0);
                float v11 = get_pix(y1, x1);

                float wx = src_x - x0;
                float wy = src_y - y0;
                dst(y,x) = (1-wy)*((1-wx)*v00 + wx*v10) + wy*((1-wx)*v01 + wx*v11);
            }
        }
        return dst;
    }
};
MatrixXf Engine::D = MatrixXf();
int Engine::D_N = 0;

MatrixXf resz(const MatrixXf& s, int t) {
    if (s.rows() == t && s.cols() == t) return s;
    MatrixXf d = MatrixXf::Zero(t,t);
    for(int r=0;r<t;++r) for(int c=0;c<t;++c) {
        int rr = min((int)(r * s.rows() / t), (int)s.rows()-1);
        int cc = min((int)(c * s.cols() / t), (int)s.cols()-1);
        d(r,c) = s(rr,cc);
    }
    return d;
}

// ---------------- Kahan Summation Helper ----------------
// Performs numerically stable addition: target += addend
// Uses a compensation matrix 'comp' of the same size as 'target' to track low-order bits.
void kahan_add_block(MatrixXf &target, MatrixXf &comp, const MatrixXf &addend, int r, int c) {
    int rows = addend.rows();
    int cols = addend.cols();
    
    for(int i=0; i<rows; ++i) {
        for(int j=0; j<cols; ++j) {
            // Kahan summation algorithm
            float y = addend(i,j) - comp(r+i, c+j);
            float t = target(r+i, c+j) + y;
            comp(r+i, c+j) = (t - target(r+i, c+j)) - y;
            target(r+i, c+j) = t;
        }
    }
}

// ---------------- Dictionary ----------------
struct Dict {
    vector<MatrixXf> atoms;
    vector<CryptoUtils::Feature> atom_features;

    void train(const MatrixXf &d, int bs, int maxe, float vth) {
        atoms.clear(); atom_features.clear();
        vector<pair<MatrixXf,float>> pool;

        for(int i=0;i<=d.rows()-bs;i+=bs/2) {
            for(int j=0;j<=d.cols()-bs;j+=bs/2) {
                MatrixXf b = d.block(i,j,bs,bs);
                MatrixXf c = b.array() - b.mean();
                float v = c.squaredNorm();
                if (v < vth) continue;
                
                MatrixXf cn = resz(c, CS);
                float nrm = cn.norm();
                if (nrm < 1e-6f) continue;
                cn /= nrm;
                pool.emplace_back(cn, v);
            }
        }
        
        sort(pool.begin(), pool.end(), [](auto &a, auto &b){ return a.second > b.second; });

        for(const auto &it : pool) {
            if ((int)atoms.size() >= maxe) break;
            bool redundant = false;
            for(const auto &a : atoms) {
                float sim = (a.array() * it.first.array()).sum();
                if (fabs(sim) > 0.90f) { redundant = true; break; }
            }
            if (!redundant) {
                atoms.push_back(it.first);
                atom_features.push_back(CryptoUtils::analyze(it.first));
            }
        }
    }

    void ser(stringstream &ss, bool decode_mode) {
        if (!decode_mode) {
            uint32_t n = (uint32_t)atoms.size();
            ss.write((char*)&n, 4);
            Engine::ensureD(CS);
            for (auto &a : atoms) {
                MatrixXf co = Engine::dct(a, false);
                for(int i=0;i<CS;++i) for(int j=0;j<CS;++j) {
                    double val = (double)co(i,j) * 127.0 / (1.0 + (i+j)*0.5);
                    int8_t v = (int8_t)clamp(val, -128.0, 127.0);
                    ss.write((char*)&v, 1);
                }
            }
        } else {
            uint32_t n = 0; ss.read((char*)&n, 4);
            atoms.assign(n, MatrixXf::Zero(CS,CS));
            atom_features.clear();
            Engine::ensureD(CS);
            for(uint32_t k=0;k<n;++k) {
                MatrixXf co(CS,CS);
                for(int i=0;i<CS;++i) for(int j=0;j<CS;++j) {
                    int8_t v; ss.read((char*)&v, 1);
                    co(i,j) = (float)((double)v * (1.0 + (i+j)*0.5) / 127.0);
                }
                atoms[k] = Engine::dct(co, true);
                atom_features.push_back(CryptoUtils::analyze(atoms[k]));
            }
        }
    }

    Entry solve(const MatrixXf &t, float &sc_out) {
        Entry e; sc_out = 0.f;
        if (atoms.empty()) return e;
        
        MatrixXf tc = resz(t, CS);
        float tn = tc.norm();
        if (tn < 1e-6f) return e;
        tc /= tn;

        CryptoUtils::Feature ft = CryptoUtils::analyze(tc);

        float best_score = -FLT_MAX;
        int best_id = 0;
        float best_ang = 0;
        float best_sx = 0, best_sy = 0;
        bool best_flip = false;
        bool best_neg = false;

        for(size_t i=0; i<atoms.size(); ++i) {
            const auto& fa = atom_features[i];
            float d_ang = CryptoUtils::solve_delta_angle(ft.angle, fa.angle);
            
            float angles_to_try[] = { d_ang, d_ang + (float)PI };
            
            for(float ang : angles_to_try) {
                float ca = cos(ang), sa = sin(ang);
                float rx = fa.center.x() * ca - fa.center.y() * sa;
                float ry = fa.center.x() * sa + fa.center.y() * ca;
                float sx = ft.center.x() - rx;
                float sy = ft.center.y() - ry;

                MatrixXf cand = Engine::warp(atoms[i], ang, sx, sy, false);
                float sim = (cand.array() * tc.array()).sum();
                
                MatrixXf cand_f = Engine::warp(atoms[i], ang, sx, sy, true);
                float sim_f = (cand_f.array() * tc.array()).sum();

                float abs_sim = fabs(sim);
                float abs_sim_f = fabs(sim_f);

                if (abs_sim > best_score) {
                    best_score = abs_sim;
                    best_id = i; best_ang = ang; best_sx = sx; best_sy = sy;
                    best_flip = false; best_neg = (sim < 0);
                }
                if (abs_sim_f > best_score) {
                    best_score = abs_sim_f;
                    best_id = i; best_ang = ang; best_sx = sx; best_sy = sy;
                    best_flip = true; best_neg = (sim_f < 0);
                }
            }
        }

        e.id = (uint16_t)best_id;
        while(best_ang < 0) best_ang += 2*PI;
        e.ang = (uint8_t)( (best_ang / (2*PI)) * Q_ANG ) % Q_ANG;
        
        auto q_shf = [](float v) { return (uint8_t)clamp((v + 4.0f) * 2.0f, 0.0f, 15.0f); };
        e.sx = q_shf(best_sx);
        e.sy = q_shf(best_sy);
        e.flg = (best_flip ? 1 : 0) | (best_neg ? 2 : 0);
        
        sc_out = best_score * tn;
        if (best_neg) sc_out = -sc_out;
        return e;
    }
    
    MatrixXf reconstruct(const Entry &e) {
        if (e.id >= atoms.size()) return MatrixXf::Zero(CS,CS);
        float ang = (float)e.ang / Q_ANG * 2.0f * PI;
        float sx = ((float)e.sx / 2.0f) - 4.0f;
        float sy = ((float)e.sy / 2.0f) - 4.0f;
        bool flip = (e.flg & 1);
        bool neg  = (e.flg & 2);
        
        MatrixXf m = Engine::warp(atoms[e.id], ang, sx, sy, flip);
        if (neg) m = -m;
        return m;
    }
};

// ---------------- RDO and codec ----------------
struct RDO { float ssd = FLT_MAX; float bits = 0; string s; };

RDO comp_qt(const MatrixXf &src, int r, int c, int sz, Dict &d, MatrixXf &rec, float lambda, bool split, int ms) {
    RDO res;
    int he = min(sz, (int)src.rows() - r), we = min(sz, (int)src.cols() - c);
    if (he <= 0 || we <= 0) return res;

    MatrixXf blk = src.block(r,c,he,we);
    float mu = blk.mean();
    float sc = 0.f;
    float qs = 2.0f;

    Entry e = d.solve(blk.array() - mu, sc);
    // FIX 1: Direct DC offset storage (0..255) instead of biased +128
    e.off = (uint8_t)clamp(mu, 0.f, 255.f); 
    e.gn = (uint8_t)clamp(fabs(sc) / qs, 0.f, 255.f);

    MatrixXf at = MatrixXf::Zero(sz, sz);
    if (e.id < d.atoms.size() && e.gn > 0) {
        MatrixXf a = d.reconstruct(e);
        at = resz(a, sz);
        float an = at.norm();
        if (an > 1e-6f) at *= (e.gn * qs / an);
    }

    // FIX 1 cont: Reconstruct using direct offset (no -128)
    MatrixXf lr = (at.array() + (float)e.off).matrix().block(0,0,he,we);
    float cost = (blk - lr).squaredNorm() + lambda * (sizeof(Entry) * 8.0f);

    if (split && sz > ms && sz >= 4) {
        int h = sz/2;
        RDO ch[4];
        float ss = 0, sb = 4;
        for(int i=0;i<4;++i) {
            ch[i] = comp_qt(src, r + (i/2)*h, c + (i%2)*h, h, d, rec, lambda, true, ms);
            ss += ch[i].ssd; sb += ch[i].bits;
        }
        if (ss + lambda * sb < cost) {
            res.ssd = ss; res.bits = sb;
            uint8_t f = QS; res.s.append((char*)&f,1);
            for(int i=0;i<4;++i) res.s.append(ch[i].s);
            return res;
        }
    }

    rec.block(r,c,he,we) = lr;
    res.ssd = (blk - lr).squaredNorm();
    res.bits = sizeof(Entry) * 8.0f;
    if (split) { uint8_t f = QL; res.s.append((char*)&f,1); }
    res.s.append((char*)&e, sizeof(Entry));
    return res;
}

// Fixed dec_qt with Kahan summation support
void dec_qt(stringstream &ss, MatrixXf &t, MatrixXf &kahan_c, int r, int c, int sz, Dict &d, bool split) {
    if (split) {
        uint8_t f = 0; if (!ss.read((char*)&f,1)) return;
        if (f == QS) {
            int h = sz/2;
            for(int i=0;i<4;++i) dec_qt(ss, t, kahan_c, r + (i/2)*h, c + (i%2)*h, h, d, true);
            return;
        }
    }
    Entry e; if (!ss.read((char*)&e, sizeof(Entry))) return;
    int he = min(sz, (int)t.rows() - r), we = min(sz, (int)t.cols() - c);
    
    // FIX 1 cont: Decode offset directly
    MatrixXf at = MatrixXf::Constant(he,we, (float)e.off);
    if (e.id < d.atoms.size() && e.gn > 0) {
        MatrixXf a = resz(d.reconstruct(e), sz);
        float an = a.norm();
        if (an > 1e-6f) at += (a.array() * (e.gn * 2.0f / an)).matrix().block(0,0,he,we);
    }
    
    // Kahan Summation: t += at
    kahan_add_block(t, kahan_c, at, r, c);
}

// ---------------- Process ----------------
void process(const string &in, const string &out, bool decode_mode, const Config &cfg = Config()) {
    stringstream bs;
    if (!decode_mode) {
        ifstream fin(in, ios::binary);
        if (!fin) return;
        string magic; int w,h,maxv;
        fin >> magic >> w >> h >> maxv; fin.ignore(1);
        if (magic != "P6") return;

        uint32_t mag = 0x47484558; bs.write((char*)&mag,4);
        bs.write((char*)&w,4); bs.write((char*)&h,4);
        bs.write((char*)&cfg.bs,4); bs.write((char*)&cfg.rs,4);

        vector<MatrixXf> ch(3, MatrixXf::Zero(h,w));
        for(int i=0;i<h;i++) for(int j=0;j<w;j++) for(int k=0;k<3;k++) {
            uint8_t v; fin.read((char*)&v,1); ch[k](i,j) = v;
        }

        for(int k=0;k<3;k++) {
            Dict bd, rd;
            MatrixXf rec = MatrixXf::Zero(h,w);
            
            // Base Layer
            bd.train(ch[k], cfg.bs, cfg.be, cfg.bvt);
            bd.ser(bs, false);
            for(int i=0;i<h;i+=cfg.bs) for(int j=0;j<w;j+=cfg.bs) {
                RDO r = comp_qt(ch[k], i, j, cfg.bs, bd, rec, cfg.lb, true, cfg.bs/2);
                uint32_t s = (uint32_t)r.s.size(); bs.write((char*)&s,4);
                if(s) bs.write(r.s.data(), s);
            }
            
            // Residual Layer
            MatrixXf resi = ch[k] - rec;
            rd.train(resi, cfg.rs, cfg.re, cfg.rvt);
            rd.ser(bs, false);
            for(int i=0;i<h;i+=cfg.rs) for(int j=0;j<w;j+=cfg.rs) {
                float sc=0;
                Entry e = rd.solve(resi.block(i,j,min(cfg.rs,h-i), min(cfg.rs,w-j)), sc);
                e.gn = (uint8_t)clamp(fabs(sc)/1.5f, 0.f, 255.f);
                if(sc<0) e.flg |= 2;
                e.off = 0; // FIX 2: Explicitly zero out residual offset to prevent color shift
                bs.write((char*)&e, sizeof(Entry));
            }
        }
        
        ofstream fo(out, ios::binary);
        boost::iostreams::filtering_streambuf<boost::iostreams::output> z;
        z.push(boost::iostreams::zlib_compressor()); z.push(fo);
        boost::iostreams::copy(bs, z);
    } else {
        ifstream fi(in, ios::binary);
        if (!fi) return;
        boost::iostreams::filtering_streambuf<boost::iostreams::input> z;
        z.push(boost::iostreams::zlib_decompressor()); z.push(fi);
        boost::iostreams::copy(z, bs);

        uint32_t mag; bs.read((char*)&mag,4);
        int w,h,bsz,rsz; bs.read((char*)&w,4); bs.read((char*)&h,4); bs.read((char*)&bsz,4); bs.read((char*)&rsz,4);

        ofstream fo(out, ios::binary);
        fo << "P6\n" << w << " " << h << "\n255\n";
        vector<MatrixXf> ch(3, MatrixXf::Zero(h,w));
        // Kahan Compensation Buffers
        vector<MatrixXf> ch_kahan_comp(3, MatrixXf::Zero(h,w));

        for(int k=0;k<3;k++) {
            Dict bd, rd;
            bd.ser(bs, true);
            for(int i=0;i<h;i+=bsz) for(int j=0;j<w;j+=bsz) {
                uint32_t s; bs.read((char*)&s,4);
                string t; if(s){ t.assign(s,0); bs.read(&t[0],s); }
                stringstream ss(t);
                // Pass Kahan compensation buffer
                dec_qt(ss, ch[k], ch_kahan_comp[k], i, j, bsz, bd, true);
            }
            rd.ser(bs, true);
            for(int i=0;i<h;i+=rsz) for(int j=0;j<w;j+=rsz) {
                Entry e; bs.read((char*)&e, sizeof(Entry));
                int he = min(rsz, h-i), we = min(rsz, w-j);
                if (e.id < rd.atoms.size() && e.gn > 0) {
                    MatrixXf a = resz(rd.reconstruct(e), rsz);
                    float an = a.norm();
                    // FIX 3: Slicing bug fix from previous step included here
                    if(an>1e-6f) {
                        MatrixXf val = (a * (e.gn*1.5f/an)).block(0,0,he,we);
                        // Kahan Add for Residual
                        kahan_add_block(ch[k], ch_kahan_comp[k], val, i, j);
                    }
                }
            }
        }
        for(int i=0;i<h;i++) for(int j=0;j<w;j++) for(int k=0;k<3;k++) {
            uint8_t v = (uint8_t)clamp(ch[k](i,j), 0.f, 255.f);
            fo.write((char*)&v, 1);
        }
    }
}

void help(const char* p) {
    cout << "GDHC v10.6 - Compact Spectral Hybrid\n"
         << "Usage: " << p << " <c/d> <in> <out> [opts]\n\n"
         << "Opts:\n"
         << " --bs <int>  Base block (16)\n"
         << " --rs <int>  Resid block (8)\n"
         << " --be <int>  Base entries (256)\n"
         << " --re <int>  Resid entries (512)\n"
         << " --lb <float> Base lambda (250)\n"
         << " --lr <float> Resid lambda (120)\n"
         << " --bv <float> Base var (40)\n"
         << " --rv <float> Resid var (10)\n";
}

int main(int argc, char** argv) {
    if (argc < 4) { help(argv[0]); return 1; }

    string m = argv[1], in = argv[2], out = argv[3];
    Config c;

    for (int i = 4; i < argc; ++i) {
        string a = argv[i];
        if (i + 1 >= argc) break;
        if (a == "--bs") c.bs = stoi(argv[++i]);
        else if (a == "--rs") c.rs = stoi(argv[++i]);
        else if (a == "--be") c.be = stoi(argv[++i]);
        else if (a == "--re") c.re = stoi(argv[++i]);
        else if (a == "--lb") c.lb = stof(argv[++i]);
        else if (a == "--lr") c.lr = stof(argv[++i]);
        else if (a == "--bv") c.bvt = stof(argv[++i]);
        else if (a == "--rv") c.rvt = stof(argv[++i]);
    }

    try {
        bool dec = (m == "d");
        cout << (dec ? "Decompressing..." : "Compressing...") << endl;
        process(in, out, dec, c);
        cout << "Done." << endl;
    } catch (const exception& e) {
        cerr << "Err: " << e.what() << endl;
        return 1;
    }
    return 0;
}

