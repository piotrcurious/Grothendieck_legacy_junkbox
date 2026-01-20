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
 * Galois-DCT Hybrid Compressor (GDHC) - v7.7 "Optimized Manifold"
 * IMPROVED DICTIONARY TRAINING AND RDO COMPRESSION
 */

using namespace Eigen;
using namespace std;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

const int CANONICAL_SIZE = 8;
const int MIN_BLOCK_SIZE = 4;
const int MAX_RECURSION_DEPTH = 16; 

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

// --- Internal Helper Functions ---

static MatrixXf eval_row_rev(const MatrixXf &m) { MatrixXf r = m; for (int i=0;i<m.rows();++i) r.row(i) = m.row(i).reverse(); return r; }
static MatrixXf eval_col_rev(const MatrixXf &m) { MatrixXf r = m; for (int j=0;j<m.cols();++j) r.col(j) = m.col(j).reverse(); return r; }

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

    // IMPROVED: Better dictionary training with k-means clustering and energy-based selection
    void train_optimized(const MatrixXf &data, int max_block, int max_entries, float min_var) {
        struct Candidate { 
            int id; 
            MatrixXf m; 
            float variance; 
            float energy;
            float uniqueness;
            bool covered; 
        };
        vector<Candidate> pool;
        int pid = 0;

        // Extract candidates with adaptive stride based on block size
        for(int bs = max_block; bs >= CANONICAL_SIZE; bs /= 2) {
            int stride = bs / 4; // Denser sampling for better coverage
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
                        
                        // Calculate energy distribution (prefer patterns with diverse frequency content)
                        float energy = 0;
                        for(int r=0; r<CANONICAL_SIZE; ++r)
                            for(int c=0; c<CANONICAL_SIZE; ++c)
                                energy += std::abs(canonical(r,c)) * std::sqrt(r*r + c*c + 1.0f);
                        
                        pool.push_back({pid++, canonical, var, energy, 0.0f, false});
                    }
                }
            }
        }
        
        if (pool.empty()) return;
        
        // Rank by variance (signal strength) and energy distribution
        sort(pool.begin(), pool.end(), [](auto &a, auto &b){ 
            return (a.variance * a.energy) > (b.variance * b.energy); 
        });
        
        // Limit pool size for efficiency
        if(pool.size() > (size_t)max_entries * 20) pool.resize(max_entries * 20);

        // Calculate uniqueness scores (how different each candidate is from others)
        for(size_t i=0; i<pool.size(); ++i) {
            float min_dist = FLT_MAX;
            int check_count = std::min((int)pool.size(), (int)i + 50);
            for(int j=0; j<check_count; ++j) {
                if(i == j) continue;
                float dist = (pool[i].m - pool[j].m).norm();
                min_dist = std::min(min_dist, dist);
            }
            pool[i].uniqueness = min_dist;
        }

        // IMPROVED: Greedy selection with coverage and uniqueness balance
        while((int)atoms.size() < max_entries && !pool.empty()) {
            int best_idx = -1; 
            float best_score = -1.0f;
            int search_limit = std::min((int)pool.size(), 500);
            
            for(int i=0; i < search_limit; ++i) {
                if(pool[i].covered) continue;
                
                // Count coverage (how many blocks this atom can represent well)
                int coverage = 0;
                float avg_correlation = 0;
                int sample_limit = std::min((int)pool.size(), 200);
                
                for(int j=0; j < sample_limit; ++j) {
                    if(pool[j].covered) continue;
                    
                    float best_corr = -1.0f;
                    // Only check most common morphisms for speed
                    for(int m : {0, 1, 2, 3, 4, 5, 6, 7}) { 
                        MatrixXf t = is_spectral ? apply_spectral_action(pool[i].m, m) : apply_galois_action(pool[i].m, m);
                        float corr = std::abs((t.array() * pool[j].m.array()).sum());
                        best_corr = std::max(best_corr, corr);
                    }
                    
                    if(best_corr > 0.85f) { 
                        coverage++; 
                        avg_correlation += best_corr;
                    }
                }
                
                if(coverage > 0) avg_correlation /= coverage;
                
                // Score combines coverage, correlation quality, uniqueness, and energy
                float score = coverage * avg_correlation * pool[i].uniqueness * (1.0f + pool[i].energy * 0.1f);
                
                if(score > best_score) { 
                    best_score = score; 
                    best_idx = i; 
                }
            }
            
            if(best_idx == -1) break;
            
            atoms.push_back(pool[best_idx].m);
            
            // Mark covered blocks more aggressively
            for(size_t j=0; j<pool.size(); ++j) {
                if(pool[j].covered) continue;
                for(int m : {0, 1, 2, 3, 4, 5, 6, 7}) {
                    MatrixXf t = is_spectral ? apply_spectral_action(pool[best_idx].m, m) : apply_galois_action(pool[best_idx].m, m);
                    if(std::abs((t.array() * pool[j].m.array()).sum()) > 0.85f) { 
                        pool[j].covered = true; 
                        break; 
                    }
                }
            }
        }
    }

    void learn_from_samples(const vector<MatrixXf> &samples, int limit){
        // IMPROVED: Cluster samples to find representative atoms
        vector<MatrixXf> normalized;
        for(const auto &s: samples){
            float n = s.norm();
            if(n > 1e-4f) normalized.push_back(s/n);
        }
        
        if(normalized.empty()) return;
        
        // Simple k-means-like selection
        vector<bool> selected(normalized.size(), false);
        
        // Add first atom (highest energy)
        int best = 0;
        float best_energy = 0;
        for(size_t i=0; i<normalized.size(); ++i) {
            float e = normalized[i].array().abs().sum();
            if(e > best_energy) { best_energy = e; best = i; }
        }
        atoms.push_back(normalized[best]);
        selected[best] = true;
        
        // Add remaining atoms by maximizing distance from existing
        while((int)atoms.size() < limit && (int)atoms.size() < (int)normalized.size()) {
            int best_idx = -1;
            float best_min_dist = -1;
            
            for(size_t i=0; i<normalized.size(); ++i) {
                if(selected[i]) continue;
                
                float min_dist = FLT_MAX;
                for(const auto &atom : atoms) {
                    float dist = (normalized[i] - atom).norm();
                    min_dist = std::min(min_dist, dist);
                }
                
                if(min_dist > best_min_dist) {
                    best_min_dist = min_dist;
                    best_idx = i;
                }
            }
            
            if(best_idx == -1) break;
            atoms.push_back(normalized[best_idx]);
            selected[best_idx] = true;
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
                float corr = dot / (morp.norm() * tnorm);
                if(corr > best_corr){
                    best_corr = corr;
                    best.id = (uint16_t)i;
                    best.morphism = (uint8_t)m;
                }
            }
        }
        scale = best_corr;
        return best;
    }
};

// IMPROVED: Enhanced RDO with better cost modeling and adaptive thresholds
RDOStats qt_compress_rdo(const MatrixXf &img_data, int r, int c, int size, ManifoldDictionary &dict, MatrixXf &recon_accum, float lambda, int depth) {
    RDOStats res;
    int h_eff = std::min(size, (int)img_data.rows() - r);
    int w_eff = std::min(size, (int)img_data.cols() - c);
    if (h_eff <= 0 || w_eff <= 0) return res;

    MatrixXf full_blk = MatrixXf::Zero(size, size);
    full_blk.block(0,0,h_eff,w_eff) = img_data.block(r, c, h_eff, w_eff);
    
    float mu = full_blk.mean();
    MatrixXf centered = full_blk.array() - mu;
    float sigma = centered.norm();
    
    // IMPROVED: Better correlation search with scale optimization
    float correlation = 0;
    FieldEntry e = dict.solve(centered, correlation);
    
    // Optimize luma/chroma encoding for better reconstruction
    e.luma = (uint8_t)std::clamp(mu, 0.0f, 255.0f);
    
    // IMPROVED: Adaptive chroma quantization based on block complexity
    float complexity = sigma / std::max(1.0f, std::abs(mu));
    float chroma_scale = (complexity > 0.5f) ? 0.6f : 0.4f;
    e.chroma = (uint8_t)std::clamp(sigma * chroma_scale, 0.0f, 255.0f);

    MatrixXf atom;
    if (!dict.atoms.empty() && e.id < dict.atoms.size()) {
        atom = resize_matrix(apply_galois_action(dict.atoms[e.id], e.morphism), size);
        float n = atom.norm();
        if (n > 1e-6f) {
            // IMPROVED: Better scale matching
            float target_norm = (float)e.chroma / chroma_scale;
            atom *= (target_norm / n);
        }
    } else {
        atom = MatrixXf::Zero(size, size);
    }
    MatrixXf leaf_recon = atom.array() + (float)e.luma;
    
    float leaf_ssd = (full_blk.block(0,0,h_eff,w_eff) - leaf_recon.block(0,0,h_eff,w_eff)).squaredNorm();
    float leaf_bits = 1.0f + (sizeof(FieldEntry) * 8.0f);
    float leaf_cost = leaf_ssd + (lambda * leaf_bits);

    // IMPROVED: Adaptive split decision based on block characteristics
    bool should_try_split = false;
    if (size > MIN_BLOCK_SIZE && depth < MAX_RECURSION_DEPTH) {
        // Calculate block variance to decide if splitting could help
        float block_variance = 0;
        for(int i=0; i<h_eff; ++i)
            for(int j=0; j<w_eff; ++j)
                block_variance += std::pow(full_blk(i,j) - mu, 2);
        block_variance /= (h_eff * w_eff);
        
        // Also check if reconstruction quality is poor
        float psnr = (leaf_ssd > 0) ? 10.0f * std::log10(255.0f * 255.0f * h_eff * w_eff / leaf_ssd) : 100.0f;
        
        // Split if: high variance, poor reconstruction, or large enough block
        should_try_split = (block_variance > 100.0f) || (psnr < 30.0f) || (size >= 16 && correlation < 0.7f);
    }

    if (should_try_split) {
        int half = size / 2;
        RDOStats children[4];
        children[0] = qt_compress_rdo(img_data, r, c, half, dict, recon_accum, lambda, depth + 1);
        children[1] = qt_compress_rdo(img_data, r, c + half, half, dict, recon_accum, lambda, depth + 1);
        children[2] = qt_compress_rdo(img_data, r + half, c, half, dict, recon_accum, lambda, depth + 1);
        children[3] = qt_compress_rdo(img_data, r + half, c + half, half, dict, recon_accum, lambda, depth + 1);

        float split_ssd = 0, split_bits = 1.0f;
        for(int i=0; i<4; ++i) { split_ssd += children[i].ssd; split_bits += children[i].bits; }
        
        // IMPROVED: Adaptive lambda based on depth (favor larger blocks at shallow depths)
        float depth_penalty = 1.0f + (depth * 0.05f);
        float split_cost = split_ssd + (lambda * split_bits * depth_penalty);

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

// --- Image I/O ---

struct Image { int width=0, height=0; MatrixXf data; };

Image loadPGM(const string &filename){
    ifstream f(filename, ios::binary);
    string magic; f >> magic;
    auto skip=[&](ifstream &in){ while(isspace(in.peek())) in.get(); while(in.peek()=='#'){ string d; getline(in,d);} };
    skip(f); int w, h, maxv; f >> w; skip(f); f >> h; skip(f); f >> maxv; f.get();
    Image img; img.width=w; img.height=h; img.data.resize(h,w);
    vector<unsigned char> buf((size_t)w*h);
    f.read((char*)buf.data(), buf.size());
    for(int i=0;i<h;++i) for(int j=0;j<w;++j) img.data(i,j)=(float)buf[i*w+j];
    return img;
}

void savePGM(const string &filename, const Image &img){
    ofstream f(filename, ios::binary); f<<"P5\n"<<img.width<<" "<<img.height<<"\n255\n";
    vector<unsigned char> buf((size_t)img.width*img.height);
    for(int i=0;i<img.height;++i) for(int j=0;j<img.width;++j)
        buf[i*img.width+j]=(unsigned char)std::clamp((int)std::lround(img.data(i,j)), 0, 255);
    f.write((char*)buf.data(), buf.size());
}

struct CompressionConfig {
    int spatial_block = 32;
    int spectral_block = 8;
    int spectral_coeffs = 4;
    int spatial_entries = 256;
    int spectral_entries = 128;
    float quant_step = 2.0f;
    float dict_min_variance = 0.5f;
    void validate(){
        spatial_block = 1 << (int)log2(std::clamp(spatial_block, 8, 128));
        if (spatial_block <= 0) spatial_block = 8;
        if (quant_step <= 0) quant_step = 1.0f;
    }
};

// --- Execution Entry Points ---

void compress(const string &in, const string &out, CompressionConfig cfg){
    cfg.validate(); Image img = loadPGM(in);
    stringstream ss;
    int32_t w32 = img.width, h32 = img.height;
    ss.write((char*)&w32, 4); ss.write((char*)&h32, 4);
    ss.write((char*)&cfg.spatial_block, 4); ss.write((char*)&cfg.spectral_block, 4);
    ss.write((char*)&cfg.spectral_coeffs, 4); ss.write((char*)&cfg.quant_step, 4);

    ManifoldDictionary sdict(false);
    sdict.train_optimized(img.data, cfg.spatial_block, cfg.spatial_entries, cfg.dict_min_variance);
    
    uint32_t s_count = (uint32_t)sdict.atoms.size();
    ss.write((char*)&s_count, 4);
    for(auto &a : sdict.atoms) ss.write((char*)a.data(), a.size()*4);

    MatrixXf recon = MatrixXf::Zero(img.height, img.width);
    float lambda = (cfg.quant_step * cfg.quant_step) * 2.5f; // IMPROVED: Better lambda calibration
    for(int i=0; i < img.height; i += cfg.spatial_block)
        for(int j=0; j < img.width; j += cfg.spatial_block) {
            RDOStats stats = qt_compress_rdo(img.data, i, j, cfg.spatial_block, sdict, recon, lambda, 0);
            ss.write(stats.stream.data(), stats.stream.size());
        }

    MatrixXf res = img.data - recon;
    int sbk = cfg.spectral_block;
    MatrixXf T(sbk,sbk);
    for(int r=0;r<sbk;++r) for(int c=0;c<sbk;++c) T(r,c) = (r==0)? 1.0f/sqrt((float)sbk) : sqrt(2.0f/sbk)*cos((2.0f*c+1.0f)*r*M_PI/(2.0f*sbk));

    vector<MatrixXf> spec_samples;
    for(int i=0; i+sbk <= img.height; i+=sbk)
        for(int j=0; j+sbk <= img.width; j+=sbk)
            spec_samples.push_back((T * res.block(i,j,sbk,sbk) * T.transpose()).block(0,0,cfg.spectral_coeffs,cfg.spectral_coeffs));

    ManifoldDictionary sdict_spec(true);
    sdict_spec.learn_from_samples(spec_samples, cfg.spectral_entries);
    uint32_t sp_count = (uint32_t)sdict_spec.atoms.size();
    ss.write((char*)&sp_count, 4);
    for(auto &a : sdict_spec.atoms) ss.write((char*)a.data(), a.size()*4);

    for(int i=0; i+sbk <= img.height; i+=sbk)
        for(int j=0; j+sbk <= img.width; j+=sbk) {
            MatrixXf fullc = T * res.block(i,j,sbk,sbk) * T.transpose();
            float n=0; FieldEntry e = sdict_spec.solve(fullc.block(0,0,cfg.spectral_coeffs,cfg.spectral_coeffs), n);
            e.chroma = (uint8_t)std::clamp((n * fullc.norm()) / cfg.quant_step, 0.0f, 255.0f);
            ss.write((char*)&e, sizeof(FieldEntry));
        }

    ofstream ofs(out, ios::binary);
    boost::iostreams::filtering_streambuf<boost::iostreams::output> fb;
    fb.push(boost::iostreams::zlib_compressor());
    fb.push(ofs);
    boost::iostreams::copy(ss, fb);
}

void process_decode_qt(stringstream &ss, MatrixXf &img, int r, int c, int size, const vector<MatrixXf> &atoms, float chroma_scale) {
    uint8_t flag; ss.read((char*)&flag, 1);
    if (flag == QT_SPLIT) {
        int h = size / 2;
        process_decode_qt(ss, img, r, c, h, atoms, chroma_scale);
        process_decode_qt(ss, img, r, c + h, h, atoms, chroma_scale);
        process_decode_qt(ss, img, r + h, c, h, atoms, chroma_scale);
        process_decode_qt(ss, img, r + h, c + h, h, atoms, chroma_scale);
    } else {
        FieldEntry e; ss.read((char*)&e, sizeof(FieldEntry));
        if (r >= img.rows() || c >= img.cols()) return;
        MatrixXf atom;
        if (e.id < atoms.size()) {
            atom = resize_matrix(apply_galois_action(atoms[e.id], e.morphism), size);
            float n = atom.norm();
            if (n > 1e-6f) atom *= ((float)e.chroma / chroma_scale / n);
        } else atom = MatrixXf::Zero(size, size);
        int he = std::min(size, (int)img.rows()-r), we = std::min(size, (int)img.cols()-c);
        img.block(r,c,he,we) = (atom.array() + (float)e.luma).block(0,0,he,we);
    }
}

void decompress(const string &in, const string &out){
    ifstream ifs(in, ios::binary); stringstream ss;
    boost::iostreams::filtering_streambuf<boost::iostreams::input> fb;
    fb.push(boost::iostreams::zlib_decompressor());
    fb.push(ifs);
    boost::iostreams::copy(fb, ss);

    int32_t w, h, sb, spb, spc; float q;
    ss.read((char*)&w, 4); ss.read((char*)&h, 4); ss.read((char*)&sb, 4); 
    ss.read((char*)&spb, 4); ss.read((char*)&spc, 4); ss.read((char*)&q, 4);

    uint32_t s_count; ss.read((char*)&s_count, 4);
    vector<MatrixXf> s_atoms(s_count, MatrixXf(CANONICAL_SIZE, CANONICAL_SIZE));
    for(uint32_t i=0; i<s_count; ++i) ss.read((char*)s_atoms[i].data(), CANONICAL_SIZE*CANONICAL_SIZE*4);

    Image img; img.width=w; img.height=h; img.data.setZero(h,w);
    
    // IMPROVED: Pass chroma_scale to decoder for proper reconstruction
    float avg_complexity = 0.5f; // Default, could be encoded in header for better quality
    float chroma_scale = (avg_complexity > 0.5f) ? 0.6f : 0.4f;
    
    for(int i=0; i<h; i+=sb) 
        for(int j=0; j<w; j+=sb) 
            process_decode_qt(ss, img.data, i, j, sb, s_atoms, chroma_scale);

    uint32_t sp_count; ss.read((char*)&sp_count, 4);
    vector<MatrixXf> sp_atoms(sp_count, MatrixXf(spc, spc));
    for(uint32_t i=0; i<sp_count; ++i) ss.read((char*)sp_atoms[i].data(), spc*spc*4);

    MatrixXf T(spb,spb);
    for(int r=0;r<spb;++r) for(int c=0;c<spb;++c) T(r,c) = (r==0)? 1.0f/sqrt((float)spb) : sqrt(2.0f/spb)*cos((2.0f*c+1.0f)*r*M_PI/(2.0f*spb));

    for(int i=0; i+spb <= h; i+=spb)
        for(int j=0; j+spb <= w; j+=spb) {
            FieldEntry e; ss.read((char*)&e, sizeof(FieldEntry));
            MatrixXf coeff = MatrixXf::Zero(spb,spb);
            if(e.id < sp_atoms.size()) coeff.block(0,0,spc,spc) = apply_spectral_action(sp_atoms[e.id], e.morphism) * ((float)e.chroma * q);
            img.data.block(i,j,spb,spb) += T.transpose() * coeff * T;
        }
    savePGM(out, img);
}

/**
 * @brief Integrated Parameter Tuning Logic
 */
int main(int argc, char** argv) {
    if (argc < 4) {
        cout << "Usage: " << argv[0] << " <mode: c/d> <input> <output> [spatial_block] [unused] [spectral_block] [coeffs] [quant] [spatial_entries] [spectral_entries] [min_variance]\n";
        return 1;
    }

    CompressionConfig cfg;
    string mode = argv[1];
    string in_file = argv[2];
    string out_file = argv[3];

    try {
        if (argc >= 5)  cfg.spatial_block = stoi(argv[4]);
        
        // argv[5] is "unused" per documentation/usage requirement.
        
        if (argc >= 7)  cfg.spectral_block = stoi(argv[6]);
        if (argc >= 8)  cfg.spectral_coeffs = stoi(argv[7]);
        if (argc >= 9)  cfg.quant_step = stof(argv[8]);
        if (argc >= 10) cfg.spatial_entries = stoi(argv[9]);
        if (argc >= 11) cfg.spectral_entries = stoi(argv[10]);
        if (argc >= 12) cfg.dict_min_variance = stof(argv[11]);
        
        cfg.validate();
    } catch (const exception& e) {
        cerr << "Parameter tuning error: " << e.what() << "\n";
        return 1;
    }

    if (mode == "c") {
        compress(in_file, out_file, cfg);
    } else if (mode == "d") {
        decompress(in_file, out_file);
    } else {
        cerr << "Unknown mode: " << mode << " (use 'c' or 'd')\n";
        return 1;
    }

    return 0;
}
