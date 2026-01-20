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

/**
 * Galois-DCT Hybrid Compressor (GDHC) - v8.1 "Unified Manifold"
 * * ARCHITECTURAL RESTORATIONS & IMPROVEMENTS:
 * 1. Full Morphism Search: solve() now iterates through 32 states (16 transforms * 2 sign states).
 * 2. Negative Morphism Support: Integrated NEG_BIT (16) for handling negative correlations.
 * 3. Symmetry-Aware Training: Dictionary training now uses all 32 morphisms for coverage checks.
 * 4. Energy-Corrected Chroma: Scale from solve() is used to normalize energy during reconstruction.
 */

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

struct RDOStats {
    float ssd = FLT_MAX;
    float bits = 0;
    std::string stream;
};

#pragma pack(push,1)
struct FieldEntry { 
    uint16_t id = 0; 
    uint8_t morphism = 0; 
    uint8_t luma = 0; 
    uint8_t chroma = 0; 
};
#pragma pack(pop)

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
        if(pool.size() > 30000) pool.resize(30000);

        while((int)atoms.size() < max_entries) {
            int best_idx = -1; 
            int max_cov = -1;
            int search_limit = std::min((int)pool.size(), 3000);
            
            for(int i=0; i < search_limit; ++i) {
                if(pool[i].covered) continue;
                int cov = 0;
                float base_threshold = 0.85f + (pool[i].variance / 1000.0f) * 0.05f;
                base_threshold = std::min(base_threshold, 0.92f);
                
                for(int j=0; j < (int)pool.size(); ++j) {
                    if(pool[j].covered) continue;
                    // RESTORED: Checking all 32 morphisms for coverage during training
                    for(int m=0; m < 32; ++m) { 
                        MatrixXf t = is_spectral ? apply_spectral_action(pool[i].m, m) : 
                                                   apply_galois_action(pool[i].m, m);
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
                    MatrixXf t = is_spectral ? apply_spectral_action(pool[best_idx].m, m) : 
                                               apply_galois_action(pool[best_idx].m, m);
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
                    MatrixXf morp = is_spectral ? apply_spectral_action(a, (uint8_t)m) : 
                                                  apply_galois_action(a, (uint8_t)m);
                    float dot = (morp.array() * c.m.array()).sum(); 
                    if(std::abs(dot) > 0.88f){ exists = true; break; } 
                }
                if(exists) break; 
            }
            if(!exists) atoms.push_back(c.m); 
        }
    }

    FieldEntry solve(const MatrixXf &target, float &scale) {
        FieldEntry best; 
        scale = 0.0f;
        if(atoms.empty()) return best;
        
        MatrixXf t_canon = is_spectral ? target : resize_matrix(target, CANONICAL_SIZE);
        float tnorm = t_canon.norm();
        if(tnorm < 1e-9f) return best;

        float best_corr = -2.0f;
        
        // RESTORED: Comprehensive 32-state Morphism Search
        for(size_t i=0; i < atoms.size(); ++i){
            for(int m=0; m < 32; ++m){
                MatrixXf morp = is_spectral ? apply_spectral_action(atoms[i], m) : 
                                              apply_galois_action(atoms[i], m);
                float mnorm = morp.norm();
                if(mnorm < 1e-9f) continue;
                
                float dot = (morp.array() * t_canon.array()).sum();
                float corr = dot / (mnorm * tnorm);
                
                if(corr > best_corr){
                    best_corr = corr;
                    best.id = (uint16_t)i;
                    best.morphism = (uint8_t)m;
                }
            }
        }
        scale = best_corr; // Correlation coefficient
        return best;
    }
};

// --- RDO Compression ---

RDOStats qt_compress_rdo(const MatrixXf &img_data, int r, int c, int size, 
                         ManifoldDictionary &dict, MatrixXf &recon_accum, 
                         float lambda, int depth) {
    RDOStats res;
    int h_eff = std::min(size, (int)img_data.rows() - r);
    int w_eff = std::min(size, (int)img_data.cols() - c);
    if (h_eff <= 0 || w_eff <= 0) return res;

    MatrixXf full_blk = MatrixXf::Zero(size, size);
    full_blk.block(0,0,h_eff,w_eff) = img_data.block(r, c, h_eff, w_eff);
    
    float mu = full_blk.mean();
    MatrixXf centered = full_blk.array() - mu;
    float sigma = centered.norm();
    
    float correlation = 0;
    FieldEntry e = dict.solve(centered, correlation);
    e.luma = (uint8_t)std::clamp(mu, 0.0f, 255.0f);
    // Adjusted energy based on matching correlation
    e.chroma = (uint8_t)std::clamp(sigma * 0.5f * correlation, 0.0f, 255.0f);

    MatrixXf atom;
    if (!dict.atoms.empty() && e.id < dict.atoms.size()) {
        atom = resize_matrix(apply_galois_action(dict.atoms[e.id], e.morphism), size);
        float n = atom.norm();
        if (n > 1e-6f) atom *= ((float)e.chroma * 2.0f / n);
    } else {
        atom = MatrixXf::Zero(size, size);
    }
    MatrixXf leaf_recon = atom.array() + (float)e.luma;
    
    float leaf_ssd = (full_blk.block(0,0,h_eff,w_eff) - leaf_recon.block(0,0,h_eff,w_eff)).squaredNorm();
    float leaf_bits = 1.0f + (sizeof(FieldEntry) * 8.0f);
    float leaf_cost = leaf_ssd + (lambda * leaf_bits);

    if (size > MIN_BLOCK_SIZE && depth < MAX_RECURSION_DEPTH) {
        int half = size / 2;
        RDOStats children[4];
        children[0] = qt_compress_rdo(img_data, r, c, half, dict, recon_accum, lambda, depth + 1);
        children[1] = qt_compress_rdo(img_data, r, c + half, half, dict, recon_accum, lambda, depth + 1);
        children[2] = qt_compress_rdo(img_data, r + half, c, half, dict, recon_accum, lambda, depth + 1);
        children[3] = qt_compress_rdo(img_data, r + half, c + half, half, dict, recon_accum, lambda, depth + 1);

        float split_ssd = 0, split_bits = 1.0f;
        for(int i=0; i<4; ++i) { split_ssd += children[i].ssd; split_bits += children[i].bits; }
        float split_cost = split_ssd + (lambda * split_bits);

        if (split_cost < leaf_cost) {
            res.ssd = split_ssd; res.bits = split_bits;
            uint8_t flag = QT_SPLIT;
            res.stream.append((char*)&flag, 1);
            for(int i=0; i<4; ++i) res.stream.append(children[i].stream);
            return res;
        }
    }

    recon_accum.block(r, c, h_eff, w_eff) = leaf_recon.block(0,0,h_eff,w_eff);
    res.ssd = leaf_ssd; res.bits = leaf_bits;
    uint8_t flag = QT_LEAF;
    res.stream.append((char*)&flag, 1);
    res.stream.append((char*)&e, sizeof(FieldEntry));
    return res;
}

// --- IO and Image Handling ---

struct Image { int width=0, height=0; MatrixXf data; };

Image loadPGM(const string &filename){
    ifstream f(filename, ios::binary);
    if(!f) throw runtime_error("Cannot open input file: " + filename);
    string magic; f >> magic;
    auto skip=[&](ifstream &in){ 
        while(isspace(in.peek())) in.get(); 
        while(in.peek()=='#'){ string d; getline(in,d);} 
    };
    skip(f); int w, h, maxv; f >> w; skip(f); f >> h; skip(f); f >> maxv; f.get();
    Image img; img.width=w; img.height=h; img.data.resize(h,w);
    vector<unsigned char> buf((size_t)w*h);
    f.read((char*)buf.data(), buf.size());
    for(int i=0;i<h;++i) for(int j=0;j<w;++j) img.data(i,j)=(float)buf[i*w+j];
    return img;
}

void savePGM(const string &filename, const Image &img){
    ofstream f(filename, ios::binary); 
    f<<"P5\n"<<img.width<<" "<<img.height<<"\n255\n";
    vector<unsigned char> buf((size_t)img.width*img.height);
    for(int i=0;i<img.height;++i) for(int j=0;j<img.width;++j)
        buf[i*img.width+j]=(unsigned char)std::clamp((int)std::lround(img.data(i,j)), 0, 255);
    f.write((char*)buf.data(), buf.size());
}

struct CompressionConfig {
    int spatial_block = 32;
    int spatial_stride = 8;
    int spectral_block = 8;
    int spectral_coeffs = 4;
    int spatial_entries = 256;
    int spectral_entries = 128;
    float quant_step = 2.0f;
    float dict_min_variance = 0.5f;
    bool use_quadtree = false;
    float rdo_lambda = 6.0f;
    
    void validate(){
        spatial_block = 1 << (int)log2(std::clamp(spatial_block, 8, 128));
        if (spectral_coeffs > spectral_block) spectral_coeffs = spectral_block;
        if (spatial_stride > spatial_block) spatial_stride = spatial_block;
    }
};

void compress(const string &in, const string &out, CompressionConfig cfg){
    cfg.validate(); 
    Image img = loadPGM(in);
    stringstream ss;
    
    int32_t w32 = img.width, h32 = img.height;
    uint8_t flags = cfg.use_quadtree ? 1 : 0;
    
    ss.write((char*)&w32, 4); 
    ss.write((char*)&h32, 4);
    ss.write((char*)&flags, 1);
    ss.write((char*)&cfg.spatial_block, 4); 
    ss.write((char*)&cfg.spatial_stride, 4);
    ss.write((char*)&cfg.spectral_block, 4);
    ss.write((char*)&cfg.spectral_coeffs, 4); 
    ss.write((char*)&cfg.quant_step, 4);

    ManifoldDictionary sdict(false);
    sdict.train_optimized(img.data, cfg.spatial_block, cfg.spatial_entries, 
                         cfg.dict_min_variance, cfg.spatial_stride);
    
    uint32_t s_count = (uint32_t)sdict.atoms.size();
    ss.write((char*)&s_count, 4);
    for(auto &a : sdict.atoms) ss.write((char*)a.data(), a.size()*4);

    MatrixXf recon = MatrixXf::Zero(img.height, img.width);

    if(cfg.use_quadtree) {
        for(int i=0; i < img.height; i += cfg.spatial_block)
            for(int j=0; j < img.width; j += cfg.spatial_block) {
                RDOStats stats = qt_compress_rdo(img.data, i, j, cfg.spatial_block, 
                                                 sdict, recon, cfg.rdo_lambda, 0);
                ss.write(stats.stream.data(), stats.stream.size());
            }
    } else {
        MatrixXf weight = MatrixXf::Zero(img.height, img.width);
        for(int i=0; i+cfg.spatial_block <= img.height; i+=cfg.spatial_stride){
            for(int j=0; j+cfg.spatial_block <= img.width; j+=cfg.spatial_stride){ 
                MatrixXf blk = img.data.block(i,j,cfg.spatial_block,cfg.spatial_block);
                float mu = blk.mean(); 
                MatrixXf centered = blk.array() - mu; 
                float sigma = centered.norm(); 
                float scale; 
                FieldEntry e = sdict.solve(centered, scale);
                e.luma = (uint8_t)std::clamp(mu, 0.0f, 255.0f); 
                e.chroma = (uint8_t)std::clamp(sigma * 0.5f * scale, 0.0f, 255.0f);
                ss.write((char*)&e, sizeof(FieldEntry));
                
                MatrixXf atom;
                if(e.id < sdict.atoms.size()){ 
                    atom = resize_matrix(apply_galois_action(sdict.atoms[e.id], e.morphism), cfg.spatial_block);
                    float n = atom.norm();
                    if(n > 1e-6f) atom *= ((float)e.chroma * 2.0f / n);
                    recon.block(i,j,cfg.spatial_block,cfg.spatial_block).array() += atom.array() + (float)e.luma;
                } else { 
                    recon.block(i,j,cfg.spatial_block,cfg.spatial_block).array() += (float)e.luma; 
                }
                weight.block(i,j,cfg.spatial_block,cfg.spatial_block).array() += 1.0f;
            }
        }
        recon = recon.array() / weight.array().max(1.0f);
    }

    // Spectral residual
    MatrixXf res = img.data - recon;
    int sbk = cfg.spectral_block;
    MatrixXf T(sbk,sbk);
    for(int r=0;r<sbk;++r) for(int c=0;c<sbk;++c) 
        T(r,c) = (r==0)? 1.0f/sqrt((float)sbk) : 
                 sqrt(2.0f/sbk)*cos((2.0f*c+1.0f)*r*M_PI/(2.0f*sbk));

    vector<MatrixXf> spec_samples;
    for(int i=0; i+sbk <= img.height; i+=sbk)
        for(int j=0; j+sbk <= img.width; j+=sbk)
            spec_samples.push_back((T * res.block(i,j,sbk,sbk) * T.transpose())
                                  .block(0,0,cfg.spectral_coeffs,cfg.spectral_coeffs));

    ManifoldDictionary sdict_spec(true);
    sdict_spec.learn_from_samples(spec_samples, cfg.spectral_entries);
    
    uint32_t sp_count = (uint32_t)sdict_spec.atoms.size();
    ss.write((char*)&sp_count, 4);
    for(auto &a : sdict_spec.atoms) ss.write((char*)a.data(), a.size()*4);

    for(int i=0; i+sbk <= img.height; i+=sbk)
        for(int j=0; j+sbk <= img.width; j+=sbk) {
            MatrixXf fullc = T * res.block(i,j,sbk,sbk) * T.transpose();
            float n=0; 
            FieldEntry e = sdict_spec.solve(fullc.block(0,0,cfg.spectral_coeffs,cfg.spectral_coeffs), n);
            e.chroma = (uint8_t)std::clamp((n * fullc.block(0,0,cfg.spectral_coeffs,cfg.spectral_coeffs).norm()) / cfg.quant_step, 0.0f, 255.0f);
            ss.write((char*)&e, sizeof(FieldEntry));
        }

    ofstream ofs(out, ios::binary);
    boost::iostreams::filtering_streambuf<boost::iostreams::output> fb;
    fb.push(boost::iostreams::zlib_compressor());
    fb.push(ofs);
    boost::iostreams::copy(ss, fb);
}

void process_decode_qt(stringstream &ss, MatrixXf &img, int r, int c, int size, 
                      const vector<MatrixXf> &atoms) {
    uint8_t flag; ss.read((char*)&flag, 1);
    if (flag == QT_SPLIT) {
        int h = size / 2;
        process_decode_qt(ss, img, r, c, h, atoms);
        process_decode_qt(ss, img, r, c + h, h, atoms);
        process_decode_qt(ss, img, r + h, c, h, atoms);
        process_decode_qt(ss, img, r + h, c + h, h, atoms);
    } else {
        FieldEntry e; ss.read((char*)&e, sizeof(FieldEntry));
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

void decompress(const string &in, const string &out){
    ifstream ifs(in, ios::binary); 
    if(!ifs) throw runtime_error("Cannot open compressed file: " + in);
    stringstream ss;
    boost::iostreams::filtering_streambuf<boost::iostreams::input> fb;
    fb.push(boost::iostreams::zlib_decompressor());
    fb.push(ifs);
    boost::iostreams::copy(fb, ss);

    int32_t w, h; 
    uint8_t flags;
    CompressionConfig cfg;
    
    ss.read((char*)&w, 4); 
    ss.read((char*)&h, 4); 
    ss.read((char*)&flags, 1);
    cfg.use_quadtree = (flags & 1);
    
    ss.read((char*)&cfg.spatial_block, 4);
    ss.read((char*)&cfg.spatial_stride, 4);
    ss.read((char*)&cfg.spectral_block, 4); 
    ss.read((char*)&cfg.spectral_coeffs, 4); 
    ss.read((char*)&cfg.quant_step, 4);

    uint32_t s_count; 
    ss.read((char*)&s_count, 4);
    vector<MatrixXf> s_atoms(s_count, MatrixXf(CANONICAL_SIZE, CANONICAL_SIZE));
    for(uint32_t i=0; i<s_count; ++i) 
        ss.read((char*)s_atoms[i].data(), CANONICAL_SIZE*CANONICAL_SIZE*4);

    Image img; 
    img.width=w; 
    img.height=h; 
    img.data.setZero(h,w);
    
    if(cfg.use_quadtree) {
        for(int i=0; i<h; i+=cfg.spatial_block) 
            for(int j=0; j<w; j+=cfg.spatial_block) 
                process_decode_qt(ss, img.data, i, j, cfg.spatial_block, s_atoms);
    } else {
        MatrixXf weight = MatrixXf::Zero(h,w);
        for(int i=0; i+cfg.spatial_block <= h; i+=cfg.spatial_stride) {
            for(int j=0; j+cfg.spatial_block <= w; j+=cfg.spatial_stride){ 
                FieldEntry e; 
                ss.read((char*)&e, sizeof(FieldEntry)); 
                
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

    // Spectral decode
    uint32_t sp_count; 
    ss.read((char*)&sp_count, 4);
    vector<MatrixXf> sp_atoms(sp_count, MatrixXf(cfg.spectral_coeffs, cfg.spectral_coeffs));
    for(uint32_t i=0; i<sp_count; ++i) 
        ss.read((char*)sp_atoms[i].data(), cfg.spectral_coeffs*cfg.spectral_coeffs*4);

    int sbk = cfg.spectral_block;
    MatrixXf T(sbk,sbk);
    for(int r=0;r<sbk;++r) for(int c=0;c<sbk;++c) 
        T(r,c) = (r==0)? 1.0f/sqrt((float)sbk) : 
                 sqrt(2.0f/sbk)*cos((2.0f*c+1.0f)*r*M_PI/(2.0f*sbk));

    for(int i=0; i+sbk <= h; i+=sbk)
        for(int j=0; j+sbk <= w; j+=sbk) {
            FieldEntry e; 
            ss.read((char*)&e, sizeof(FieldEntry));
            MatrixXf coeff = MatrixXf::Zero(sbk,sbk);
            if(e.id < sp_atoms.size()) 
                coeff.block(0,0,cfg.spectral_coeffs,cfg.spectral_coeffs) = 
                    apply_spectral_action(sp_atoms[e.id], e.morphism) * ((float)e.chroma * cfg.quant_step);
            img.data.block(i,j,sbk,sbk) += T.transpose() * coeff * T;
        }
        
    savePGM(out, img);
}

int main(int argc, char** argv){
    if(argc < 4) {
        cout << "GDHC v8.1 - Unified Manifold Compressor (Full Morphism Search Enabled)\n";
        cout << "Usage: " << argv[0] << " <c|d> <input> <output> [options]\n";
        return 1;
    }
    
    CompressionConfig cfg;
    try {
        if(argc >= 5) cfg.spatial_block = stoi(argv[4]); 
        if(argc >= 6) cfg.spatial_stride = stoi(argv[5]); 
        if(argc >= 7) cfg.spectral_block = stoi(argv[6]); 
        if(argc >= 8) cfg.spectral_coeffs = stoi(argv[7]); 
        if(argc >= 9) cfg.quant_step = stof(argv[8]); 
        if(argc >= 10) cfg.spatial_entries = stoi(argv[9]);
        if(argc >= 11) cfg.spectral_entries = stoi(argv[10]);
        if(argc >= 12) cfg.dict_min_variance = stof(argv[11]);
        if(argc >= 13) cfg.use_quadtree = (stoi(argv[12]) != 0);
        if(argc >= 14) cfg.rdo_lambda = stof(argv[13]);
    } catch (...) {}

    if(argv[1][0] == 'c') {
        compress(argv[2], argv[3], cfg);
        cout << "Compression complete.\n";
    } else {
        decompress(argv[2], argv[3]);
        cout << "Decompression complete.\n";
    }
    return 0;
}
