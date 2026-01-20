#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <sstream>
#include <cstdint>
#include <cfloat>
#include <set>
#include <map>
#include <Eigen/Dense>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/zlib.hpp>

using namespace Eigen;
using namespace std;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

const int CANONICAL_SIZE = 8;
const int MIN_BLOCK_SIZE = 4;
const int MAX_RECURSION_DEPTH = 10;

enum Morphism : uint8_t {
    ID = 0, ROT90, ROT180, ROT270, FLIP_H, FLIP_V, TRANS, ANTITRANS,
    PHASE_H, PHASE_V, PHASE_D, EXP_1, EXP_2, LOG_1, SQUASH, STRETCH,
    NEG_BIT = 16
};

enum QTNodeType : uint8_t {
    QT_LEAF = 0,
    QT_SPLIT = 1
};

#pragma pack(push,1)
struct FieldEntry { 
    uint16_t id = 0; 
    uint8_t morphism = 0; 
    uint8_t luma = 0; 
    uint8_t chroma = 0; 
};
#pragma pack(pop)

struct RDOStats {
    float ssd = FLT_MAX;
    float bits = 0;
    vector<uint8_t> qt_flags;
    vector<FieldEntry> entries;
};

// --- Bitstream Helper ---
class BitWriter {
    vector<uint8_t> &buf;
    uint8_t cur_byte = 0;
    int bit_pos = 0;
public:
    BitWriter(vector<uint8_t> &b) : buf(b) {}
    void write(uint32_t val, int bits) {
        for (int i = 0; i < bits; ++i) {
            if (val & (1 << i)) cur_byte |= (1 << bit_pos);
            bit_pos++;
            if (bit_pos == 8) {
                buf.push_back(cur_byte);
                cur_byte = 0;
                bit_pos = 0;
            }
        }
    }
    void flush() {
        if (bit_pos > 0) buf.push_back(cur_byte);
        bit_pos = 0;
        cur_byte = 0;
    }
};

class BitReader {
    const vector<uint8_t> &buf;
    size_t byte_pos = 0;
    int bit_pos = 0;
public:
    BitReader(const vector<uint8_t> &b) : buf(b) {}
    uint32_t read(int bits) {
        uint32_t val = 0;
        for (int i = 0; i < bits; ++i) {
            if (byte_pos >= buf.size()) return 0;
            if (buf[byte_pos] & (1 << bit_pos)) val |= (1 << i);
            bit_pos++;
            if (bit_pos == 8) {
                bit_pos = 0;
                byte_pos++;
            }
        }
        return val;
    }
};

// --- Helper Functions ---

static MatrixXf eval_row_rev(const MatrixXf &m) { 
    MatrixXf r = m; 
    for (int i=0; i<m.rows(); ++i) r.row(i) = m.row(i).reverse(); 
    return r; 
}

static MatrixXf eval_col_rev(const MatrixXf &m) { 
    MatrixXf r = m; 
    for (int j=0; j<m.cols(); ++j) r.col(j) = m.col(j).reverse(); 
    return r; 
}

MatrixXf resize_matrix(const MatrixXf& src, int target_size) {
    if (src.rows() == target_size && src.cols() == target_size) return src;
    MatrixXf dst(target_size, target_size);
    float scale_r = (float)src.rows() / target_size;
    float scale_c = (float)src.cols() / target_size;
    for (int r = 0; r < target_size; ++r) {
        for (int c = 0; c < target_size; ++c) {
            if (scale_r >= 1.0f) {
                int sr_start = (int)(r * scale_r);
                int sr_end = std::min((int)((r + 1) * scale_r), (int)src.rows());
                int sc_start = (int)(c * scale_c);
                int sc_end = std::min((int)((c + 1) * scale_c), (int)src.cols());
                if (sr_end <= sr_start) sr_end = sr_start + 1;
                if (sc_end <= sc_start) sc_end = sc_start + 1;
                float sum = src.block(sr_start, sc_start, sr_end - sr_start, sc_end - sc_start).sum();
                dst(r,c) = sum / ((sr_end - sr_start) * (sc_end - sc_start));
            } else {
                int sr = std::min((int)(r * scale_r), (int)src.rows() - 1);
                int sc = std::min((int)(c * scale_c), (int)src.cols() - 1);
                dst(r,c) = src(sr, sc); 
            }
        }
    }
    return dst;
}

MatrixXf apply_galois_action(const MatrixXf& block, uint8_t m) {
    MatrixXf res;
    bool negate = (m & NEG_BIT);
    uint8_t op = m & 0x0F;
    if (op < 8) {
        switch(op) {
            case ROT90:     res = block.transpose().colwise().reverse().eval(); break;
            case ROT180:    res = block.reverse().eval(); break;
            case ROT270:    res = block.transpose().rowwise().reverse().eval(); break;
            case FLIP_H:    res = block.rowwise().reverse().eval(); break;
            case FLIP_V:    res = block.colwise().reverse().eval(); break;
            case TRANS:     res = block.transpose().eval(); break;
            case ANTITRANS: res = block.transpose().reverse().eval(); break;
            default:        res = block; break;
        }
    } else {
        res = block;
        switch(op) {
            case PHASE_H: res.rowwise().reverseInPlace(); break;
            case PHASE_V: res.colwise().reverseInPlace(); break;
            case EXP_1:   res = res.unaryExpr([](float v){ return std::exp(std::min(v, 3.0f)); }); break; 
            case EXP_2:   res = res.unaryExpr([](float v){ return std::exp(std::tanh(v)); }); break;
            case LOG_1:   res = res.unaryExpr([](float v){ float a = std::abs(v); return (a>1e-6f) ? std::log(a) : 0.0f; }); break;
            case SQUASH:  res = res.unaryExpr([](float v){ return 1.0f / (1.0f + std::exp(-std::clamp(v, -5.0f, 5.0f))); }); break;
            case STRETCH: res = res.unaryExpr([](float v){ return std::tanh(v); }); break;
            default: break;
        }
    }
    if (negate) res = -res;
    return res;
}

MatrixXf apply_spectral_action(const MatrixXf& coeffs, uint8_t m) {
    bool negate = (m & NEG_BIT);
    uint8_t op = m & 0x0F;
    MatrixXf res = coeffs;
    if (op & 1) res = res.transpose().eval();
    if (op & 2) res = eval_row_rev(res);
    if (op & 4) res = eval_col_rev(res);
    if (negate) res = -res;
    return res;
}

class ManifoldDictionary {
public:
    vector<MatrixXf> atoms;
    bool is_spectral;
    ManifoldDictionary(bool spec=false):is_spectral(spec){}
    void train_optimized(const MatrixXf &data, int max_block, int max_entries, float min_var, int manual_stride) {
        struct Candidate { int id; MatrixXf m; float variance; bool covered; int scale; };
        vector<Candidate> pool;
        int pid = 0;
        for(int bs = max_block; bs >= CANONICAL_SIZE; bs /= 2) {
            int stride = (manual_stride > 0) ? manual_stride : std::max(4, bs / 4); 
            for(int i=0; i + bs <= data.rows(); i += stride){
                for(int j=0; j + bs <= data.cols(); j += stride){
                    MatrixXf blk = data.block(i,j,bs,bs);
                    float mean = blk.mean();
                    MatrixXf centered = blk.array() - mean;
                    float var = centered.norm();
                    if (var < min_var) continue;
                    MatrixXf canonical = resize_matrix(centered, CANONICAL_SIZE);
                    float c_norm = canonical.norm();
                    if(c_norm > 1e-4f) {
                        canonical /= c_norm;
                        pool.push_back({pid++, canonical, var, false, bs});
                    }
                }
            }
        }
        if (pool.empty()) return;
        sort(pool.begin(), pool.end(), [](auto &a, auto &b){ return a.variance > b.variance; });
        if(pool.size() > 3000) pool.resize(3000);
        while((int)atoms.size() < max_entries) {
            int best_idx = -1; int max_cov = -1;
            int search_limit = std::min((int)pool.size(), 300);
            for(int i=0; i < search_limit; ++i) {
                if(pool[i].covered) continue;
                int cov = 0;
                float base_threshold = 0.85f + (pool[i].variance / 1000.0f) * 0.05f;
                base_threshold = std::min(base_threshold, 0.92f);
                for(int j=0; j < (int)pool.size(); ++j) {
                    if(pool[j].covered) continue;
                    for(int m=0; m < 32; ++m) { 
                        MatrixXf t = is_spectral ? apply_spectral_action(pool[i].m, m) : apply_galois_action(pool[i].m, m);
                        float dot = (t.array() * pool[j].m.array()).sum();
                        if(std::abs(dot) > base_threshold) { cov++; break; }
                    }
                }
                if(cov > max_cov) { max_cov = cov; best_idx = i; }
            }
            if(best_idx == -1) break;
            atoms.push_back(pool[best_idx].m);
            float cover_threshold = 0.85f;
            for(int j=0; j < (int)pool.size(); ++j) {
                if(pool[j].covered) continue;
                for(int m=0; m < 32; ++m) {
                    MatrixXf t = is_spectral ? apply_spectral_action(pool[best_idx].m, m) : apply_galois_action(pool[best_idx].m, m);
                    if(std::abs((t.array() * pool[j].m.array()).sum()) > cover_threshold) {
                        pool[j].covered = true;
                        break;
                    }
                }
            }
            pool[best_idx].covered = true;
        }
    }
    void learn_from_samples(const vector<MatrixXf> &samples, int limit){
        if(samples.empty() || limit <= 0) return;
        struct Node{ float score; MatrixXf m; };
        vector<Node> candidates; 
        for(const auto &s: samples){ 
            float n=s.norm(); 
            if(n > 1e-5f) candidates.push_back({n, s/n}); 
        }
        sort(candidates.begin(), candidates.end(), [](const Node&a, const Node&b){ return a.score > b.score; });
        for(const auto &c: candidates){ 
            if((int)atoms.size() >= limit) break; 
            bool exists = false; 
            for(const auto &a: atoms){ 
                if(a.rows() != c.m.rows() || a.cols() != c.m.cols()) continue;
                for(int m=0; m < 32; ++m){ 
                    MatrixXf morp = is_spectral ? apply_spectral_action(a, (uint8_t)m) : apply_galois_action(a, (uint8_t)m);
                    float dot = (morp.array() * c.m.array()).sum(); 
                    if(std::abs(dot) > 0.88f){ exists = true; break; } 
                }
                if(exists) break; 
            }
            if(!exists) atoms.push_back(c.m); 
        }
    }
    FieldEntry solve(const MatrixXf &target, float &scale) {
        FieldEntry best; scale = 0.0f;
        if(atoms.empty()) return best;
        MatrixXf t_canon = is_spectral ? target : resize_matrix(target, CANONICAL_SIZE);
        float tnorm = t_canon.norm();
        if(tnorm < 1e-9f) return best;
        float best_corr = -2.0f;
        for(size_t i=0; i < atoms.size(); ++i){
            for(int m=0; m < 32; ++m){
                MatrixXf morp = is_spectral ? apply_spectral_action(atoms[i], m) : apply_galois_action(atoms[i], m);
                float dot = (morp.array() * t_canon.array()).sum();
                if(dot > best_corr){
                    best_corr = dot;
                    best.id = (uint16_t)i;
                    best.morphism = (uint8_t)m;
                }
            }
        }
        scale = best_corr * tnorm;
        return best;
    }
};

struct Image {
    int width, height;
    MatrixXf data;
};

Image loadPGM(const string &filename) {
    ifstream ifs(filename, ios::binary);
    if (!ifs) throw runtime_error("Cannot open file: " + filename);
    string magic; ifs >> magic;
    if (magic != "P5") throw runtime_error("Not a P5 PGM file");
    int w, h, maxv; ifs >> w >> h >> maxv;
    ifs.ignore();
    Image img; img.width = w; img.height = h;
    img.data.resize(h, w);
    vector<uint8_t> buf(w * h);
    ifs.read((char*)buf.data(), w * h);
    for (int i = 0; i < h; ++i)
        for (int j = 0; j < w; ++j)
            img.data(i, j) = (float)buf[i * w + j];
    return img;
}

void savePGM(const string &filename, const Image &img) {
    ofstream ofs(filename, ios::binary);
    ofs << "P5\n" << img.width << " " << img.height << "\n255\n";
    vector<uint8_t> buf(img.width * img.height);
    for (int i = 0; i < img.height; ++i)
        for (int j = 0; j < img.width; ++j)
            buf[i * img.width + j] = (uint8_t)std::clamp(img.data(i, j), 0.0f, 255.0f);
    ofs.write((char*)buf.data(), buf.size());
}

struct CompressionConfig {
    int spatial_block = 16;
    int spatial_stride = 16;
    int spectral_block = 8;
    int spectral_coeffs = 4;
    float quant_step = 2.0f;
    int spatial_entries = 128;
    int spectral_entries = 64;
    float dict_min_variance = 10.0f;
    bool use_quadtree = true;
    float rdo_lambda = 0.1f;
};

RDOStats solve_qt(const MatrixXf &block, int r, int c, int size, int depth, ManifoldDictionary &atoms, float lambda) {
    RDOStats leaf;
    float mean = block.mean();
    MatrixXf centered = block.array() - mean;
    float scale;
    FieldEntry e = atoms.solve(centered, scale);
    e.luma = (uint8_t)std::clamp(mean, 0.0f, 255.0f);
    e.chroma = (uint8_t)std::clamp(scale / 2.0f, 0.0f, 255.0f);
    MatrixXf recon;
    if(e.id < atoms.atoms.size()){
        recon = resize_matrix(apply_galois_action(atoms.atoms[e.id], e.morphism), size);
        float n = recon.norm();
        if(n > 1e-6f) recon *= ((float)e.chroma * 2.0f / n);
    } else {
        recon = MatrixXf::Zero(size, size);
    }
    recon.array() += (float)e.luma;
    leaf.ssd = (block - recon).squaredNorm();
    leaf.bits = 1 + 31; 
    leaf.qt_flags = {QT_LEAF};
    leaf.entries = {e};
    if (size <= MIN_BLOCK_SIZE || depth >= MAX_RECURSION_DEPTH) return leaf;
    int h = size / 2;
    RDOStats split;
    split.ssd = 0; split.bits = 1;
    split.qt_flags = {QT_SPLIT};
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            RDOStats child = solve_qt(block.block(i * h, j * h, h, h), r + i * h, c + j * h, h, depth + 1, atoms, lambda);
            split.ssd += child.ssd;
            split.bits += child.bits;
            split.qt_flags.insert(split.qt_flags.end(), child.qt_flags.begin(), child.qt_flags.end());
            split.entries.insert(split.entries.end(), child.entries.begin(), child.entries.end());
        }
    }
    if (leaf.ssd + lambda * leaf.bits <= split.ssd + lambda * split.bits) return leaf;
    return split;
}

void write_entries(BitWriter &bw, const vector<FieldEntry> &entries) {
    for (const auto &e : entries) bw.write(e.id, 10);
    for (const auto &e : entries) bw.write(e.morphism, 5);
    for (const auto &e : entries) bw.write(e.luma, 8);
    for (const auto &e : entries) bw.write(e.chroma, 8);
}

void read_entries(BitReader &br, vector<FieldEntry> &entries) {
    for (auto &e : entries) e.id = br.read(10);
    for (auto &e : entries) e.morphism = br.read(5);
    for (auto &e : entries) e.luma = br.read(8);
    for (auto &e : entries) e.chroma = br.read(8);
}

void write_qt_flags(BitWriter &bw, const vector<uint8_t> &flags, int &idx) {
    uint8_t f = flags[idx++];
    bw.write(f, 1);
    if (f == QT_SPLIT) {
        for (int i = 0; i < 4; ++i) write_qt_flags(bw, flags, idx);
    }
}

void process_decode_qt(BitReader &br, MatrixXf &img, int r, int c, int size, const vector<MatrixXf> &atoms, const vector<FieldEntry> &entries, int &e_idx) {
    uint8_t flag = br.read(1);
    if (flag == QT_SPLIT) {
        int h = size / 2;
        process_decode_qt(br, img, r, c, h, atoms, entries, e_idx);
        process_decode_qt(br, img, r, c + h, h, atoms, entries, e_idx);
        process_decode_qt(br, img, r + h, c, h, atoms, entries, e_idx);
        process_decode_qt(br, img, r + h, c + h, h, atoms, entries, e_idx);
    } else {
        FieldEntry e = entries[e_idx++];
        if (r >= img.rows() || c >= img.cols()) return;
        MatrixXf atom;
        if (e.id < atoms.size()) {
            atom = resize_matrix(apply_galois_action(atoms[e.id], e.morphism), size);
            float n = atom.norm();
            if (n > 1e-6f) atom *= ((float)e.chroma * 2.0f / n);
        } else {
            atom = MatrixXf::Zero(size, size);
        }
        int he = std::min(size, (int)img.rows()-r);
        int we = std::min(size, (int)img.cols()-c);
        img.block(r,c,he,we) = (atom.array() + (float)e.luma).block(0,0,he,we);
    }
}

void compress(const string &in, const string &out, const CompressionConfig &cfg) {
    Image img = loadPGM(in);
    int w = img.width, h = img.height;
    ManifoldDictionary s_dict(false);
    s_dict.train_optimized(img.data, cfg.spatial_block, cfg.spatial_entries, cfg.dict_min_variance, cfg.spatial_stride);
    stringstream ss;
    ss.write((char*)&w, 4); ss.write((char*)&h, 4);
    uint8_t flags = (cfg.use_quadtree ? 1 : 0);
    ss.write((char*)&flags, 1);
    ss.write((char*)&cfg.spatial_block, 4); ss.write((char*)&cfg.spatial_stride, 4);
    ss.write((char*)&cfg.spectral_block, 4); ss.write((char*)&cfg.spectral_coeffs, 4);
    ss.write((char*)&cfg.quant_step, 4);
    uint32_t s_count = s_dict.atoms.size();
    ss.write((char*)&s_count, 4);
    for(const auto &a : s_dict.atoms) ss.write((char*)a.data(), CANONICAL_SIZE * CANONICAL_SIZE * 4);
    vector<uint8_t> qt_bits_buf;
    BitWriter qt_bw(qt_bits_buf);
    vector<FieldEntry> all_entries;
    if(cfg.use_quadtree) {
        for(int i=0; i<h; i+=cfg.spatial_block)
            for(int j=0; j<w; j+=cfg.spatial_block) {
                int bh = std::min(cfg.spatial_block, h-i);
                int bw = std::min(cfg.spatial_block, w-j);
                MatrixXf blk = MatrixXf::Zero(cfg.spatial_block, cfg.spatial_block);
                blk.block(0,0,bh,bw) = img.data.block(i,j,bh,bw);
                RDOStats res = solve_qt(blk, i, j, cfg.spatial_block, 0, s_dict, cfg.rdo_lambda);
                int idx = 0;
                write_qt_flags(qt_bw, res.qt_flags, idx);
                all_entries.insert(all_entries.end(), res.entries.begin(), res.entries.end());
            }
    } else {
        for(int i=0; i+cfg.spatial_block <= h; i+=cfg.spatial_stride) {
            for(int j=0; j+cfg.spatial_block <= w; j+=cfg.spatial_stride) {
                MatrixXf blk = img.data.block(i,j,cfg.spatial_block,cfg.spatial_block);
                float mean = blk.mean(); float scale;
                FieldEntry e = s_dict.solve(blk.array() - mean, scale);
                e.luma = (uint8_t)std::clamp(mean, 0.0f, 255.0f);
                e.chroma = (uint8_t)std::clamp(scale / 2.0f, 0.0f, 255.0f);
                all_entries.push_back(e);
            }
        }
    }
    qt_bw.flush();
    uint32_t qt_size = qt_bits_buf.size();
    ss.write((char*)&qt_size, 4);
    ss.write((char*)qt_bits_buf.data(), qt_size);
    uint32_t num_entries = all_entries.size();
    ss.write((char*)&num_entries, 4);
    vector<uint8_t> entry_bits_buf;
    BitWriter entry_bw(entry_bits_buf);
    write_entries(entry_bw, all_entries);
    entry_bw.flush();
    uint32_t entry_size = entry_bits_buf.size();
    ss.write((char*)&entry_size, 4);
    ss.write((char*)entry_bits_buf.data(), entry_size);
    int sbk = cfg.spectral_block;
    MatrixXf T(sbk,sbk);
    for(int r=0;r<sbk;++r) for(int c=0;c<sbk;++c) 
        T(r,c) = (r==0)? 1.0f/sqrt((float)sbk) : sqrt(2.0f/sbk)*cos((2.0f*c+1.0f)*r*M_PI/(2.0f*sbk));
    vector<MatrixXf> samples;
    for(int i=0; i+sbk <= h; i+=sbk)
        for(int j=0; j+sbk <= w; j+=sbk) {
            MatrixXf blk = img.data.block(i,j,sbk,sbk);
            MatrixXf coeff = T * blk * T.transpose();
            samples.push_back(coeff.block(0,0,cfg.spectral_coeffs,cfg.spectral_coeffs));
        }
    ManifoldDictionary sp_dict(true);
    sp_dict.learn_from_samples(samples, cfg.spectral_entries);
    uint32_t sp_count = sp_dict.atoms.size();
    ss.write((char*)&sp_count, 4);
    for(const auto &a : sp_dict.atoms) ss.write((char*)a.data(), cfg.spectral_coeffs * cfg.spectral_coeffs * 4);
    vector<FieldEntry> sp_entries;
    for(int i=0; i+sbk <= h; i+=sbk)
        for(int j=0; j+sbk <= w; j+=sbk) {
            MatrixXf blk = img.data.block(i,j,sbk,sbk);
            MatrixXf coeff = T * blk * T.transpose();
            float scale;
            FieldEntry e = sp_dict.solve(coeff.block(0,0,cfg.spectral_coeffs,cfg.spectral_coeffs), scale);
            e.luma = 0; e.chroma = (uint8_t)std::clamp(scale / cfg.quant_step, 0.0f, 255.0f);
            sp_entries.push_back(e);
        }
    uint32_t num_sp_entries = sp_entries.size();
    ss.write((char*)&num_sp_entries, 4);
    vector<uint8_t> sp_entry_bits_buf;
    BitWriter sp_entry_bw(sp_entry_bits_buf);
    write_entries(sp_entry_bw, sp_entries);
    sp_entry_bw.flush();
    uint32_t sp_entry_size = sp_entry_bits_buf.size();
    ss.write((char*)&sp_entry_size, 4);
    ss.write((char*)sp_entry_bits_buf.data(), sp_entry_size);
    ofstream ofs(out, ios::binary);
    boost::iostreams::filtering_streambuf<boost::iostreams::output> fb;
    fb.push(boost::iostreams::zlib_compressor());
    fb.push(ofs);
    boost::iostreams::copy(ss, fb);
}

void decompress(const string &in, const string &out){
    ifstream ifs(in, ios::binary); 
    if(!ifs) throw runtime_error("Cannot open compressed file: " + in);
    stringstream ss;
    boost::iostreams::filtering_streambuf<boost::iostreams::input> fb;
    fb.push(boost::iostreams::zlib_decompressor());
    fb.push(ifs);
    boost::iostreams::copy(fb, ss);
    int32_t w, h; uint8_t flags; CompressionConfig cfg;
    ss.read((char*)&w, 4); ss.read((char*)&h, 4); ss.read((char*)&flags, 1);
    cfg.use_quadtree = (flags & 1);
    ss.read((char*)&cfg.spatial_block, 4); ss.read((char*)&cfg.spatial_stride, 4);
    ss.read((char*)&cfg.spectral_block, 4); ss.read((char*)&cfg.spectral_coeffs, 4); 
    ss.read((char*)&cfg.quant_step, 4);
    uint32_t s_count; ss.read((char*)&s_count, 4);
    vector<MatrixXf> s_atoms(s_count, MatrixXf(CANONICAL_SIZE, CANONICAL_SIZE));
    for(uint32_t i=0; i<s_count; ++i) ss.read((char*)s_atoms[i].data(), CANONICAL_SIZE*CANONICAL_SIZE*4);
    uint32_t qt_size; ss.read((char*)&qt_size, 4);
    vector<uint8_t> qt_bits_buf(qt_size); ss.read((char*)qt_bits_buf.data(), qt_size);
    BitReader qt_br(qt_bits_buf);
    uint32_t num_entries; ss.read((char*)&num_entries, 4);
    uint32_t entry_size; ss.read((char*)&entry_size, 4);
    vector<uint8_t> entry_bits_buf(entry_size); ss.read((char*)entry_bits_buf.data(), entry_size);
    BitReader entry_br(entry_bits_buf);
    vector<FieldEntry> entries(num_entries);
    read_entries(entry_br, entries);
    Image img; img.width=w; img.height=h; img.data.setZero(h,w);
    int e_idx = 0;
    if(cfg.use_quadtree) {
        for(int i=0; i<h; i+=cfg.spatial_block) 
            for(int j=0; j<w; j+=cfg.spatial_block) 
                process_decode_qt(qt_br, img.data, i, j, cfg.spatial_block, s_atoms, entries, e_idx);
    } else {
        MatrixXf weight = MatrixXf::Zero(h,w);
        for(int i=0; i+cfg.spatial_block <= h; i+=cfg.spatial_stride) {
            for(int j=0; j+cfg.spatial_block <= w; j+=cfg.spatial_stride){ 
                FieldEntry e = entries[e_idx++];
                MatrixXf atom;
                if(e.id < s_atoms.size()){ 
                    atom = resize_matrix(apply_galois_action(s_atoms[e.id], e.morphism), cfg.spatial_block); 
                    float n = atom.norm();
                    if(n > 1e-6f) atom *= ((float)e.chroma*2.0f / n);
                    img.data.block(i,j,cfg.spatial_block,cfg.spatial_block).array() += atom.array() + (float)e.luma; 
                } else {
                    img.data.block(i,j,cfg.spatial_block,cfg.spatial_block).array() += (float)e.luma; 
                }
                weight.block(i,j,cfg.spatial_block,cfg.spatial_block).array() += 1.0f; 
            }
        }
        img.data = img.data.array() / weight.array().max(1.0f);
    }
    uint32_t sp_count; ss.read((char*)&sp_count, 4);
    vector<MatrixXf> sp_atoms(sp_count, MatrixXf(cfg.spectral_coeffs, cfg.spectral_coeffs));
    for(uint32_t i=0; i<sp_count; ++i) ss.read((char*)sp_atoms[i].data(), cfg.spectral_coeffs*cfg.spectral_coeffs*4);
    uint32_t num_sp_entries; ss.read((char*)&num_sp_entries, 4);
    uint32_t sp_entry_size; ss.read((char*)&sp_entry_size, 4);
    vector<uint8_t> sp_entry_bits_buf(sp_entry_size); ss.read((char*)sp_entry_bits_buf.data(), sp_entry_size);
    BitReader sp_entry_br(sp_entry_bits_buf);
    vector<FieldEntry> sp_entries(num_sp_entries);
    read_entries(sp_entry_br, sp_entries);
    int sbk = cfg.spectral_block;
    MatrixXf T(sbk,sbk);
    for(int r=0;r<sbk;++r) for(int c=0;c<sbk;++c) 
        T(r,c) = (r==0)? 1.0f/sqrt((float)sbk) : sqrt(2.0f/sbk)*cos((2.0f*c+1.0f)*r*M_PI/(2.0f*sbk));
    int sp_idx = 0;
    for(int i=0; i+sbk <= h; i+=sbk)
        for(int j=0; j+sbk <= w; j+=sbk) {
            FieldEntry e = sp_entries[sp_idx++];
            MatrixXf coeff = MatrixXf::Zero(sbk,sbk);
            if(e.id < sp_atoms.size()) 
                coeff.block(0,0,cfg.spectral_coeffs,cfg.spectral_coeffs) = 
                    apply_spectral_action(sp_atoms[e.id], e.morphism) * ((float)e.chroma * cfg.quant_step);
            img.data.block(i,j,sbk,sbk) += T.transpose() * coeff * T;
        }
    savePGM(out, img);
}

int main(int argc, char** argv){
    if(argc < 4) return 1;
    CompressionConfig cfg;
    if(argv[1][0] == 'c') { compress(argv[2], argv[3], cfg); } 
    else { decompress(argv[2], argv[3]); }
    return 0;
}
