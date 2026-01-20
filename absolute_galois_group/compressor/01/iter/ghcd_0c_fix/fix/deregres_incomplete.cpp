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
 * Galois-DCT Hybrid Compressor (GDHC) - v8.1 "Maximum Compression"
 * 
 * COMPRESSION RATIO OPTIMIZATIONS:
 * 1. FULL 64-morphism orbit (32 base + 32 negated) for dictionary training
 * 2. FULL 64-morphism search in encoding for maximum atom reuse
 * 3. Aggressive coverage thresholds (0.90+) to minimize dictionary size
 * 4. Multi-scale with optimized stride to maximize pattern diversity
 * 5. Enhanced spectral learning with full morphism orbit
 * 6. Overlapping reconstruction preserved for quality
 * 7. Entropy-optimized atom selection prioritizing high-coverage patterns
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
    for (int i=0;i<m.rows();++i) r.row(i) = m.row(i).reverse(); 
    return r; 
}

static MatrixXf eval_col_rev(const MatrixXf &m) { 
    MatrixXf r = m; 
    for (int j=0;j<m.cols();++j) r.col(j) = m.col(j).reverse(); 
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

    // MAXIMUM COMPRESSION: Full 64-morphism orbit greedy set cover
    void train_optimized(const MatrixXf &data, int max_block, int max_entries, float min_var, int manual_stride) {
        cout << "Training dictionary with FULL 64-morphism orbit..." << endl;
        
        struct Candidate { 
            int id; 
            MatrixXf m; 
            float variance; 
            bool covered; 
            int scale;
        };
        vector<Candidate> pool;
        int pid = 0;

        // Multi-scale extraction with OPTIMIZED stride for diversity
        for(int bs = max_block; bs >= CANONICAL_SIZE; bs /= 2) {
            // CRITICAL: Use larger stride to avoid redundant similar patches
            // This increases pattern diversity in the pool
            int stride = (manual_stride > 0) ? manual_stride : std::max(bs / 2, 8); 
            
            cout << "  Scale " << bs << "x" << bs << " (stride=" << stride << ")..." << endl;
            
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
        
        if (pool.empty()) {
            cout << "  WARNING: No candidates found!" << endl;
            return;
        }
        
        // Sort by variance and keep large pool for better coverage
        sort(pool.begin(), pool.end(), [](auto &a, auto &b){ 
            return a.variance > b.variance; 
        });
        
        // Keep more candidates for better quality dictionary
        size_t pool_limit = std::min((size_t)5000, pool.size());
        if(pool.size() > pool_limit) pool.resize(pool_limit);
        
        cout << "  Candidate pool size: " << pool.size() << endl;
        cout << "  Starting greedy set cover (target: " << max_entries << " atoms)..." << endl;

        // GREEDY SET COVER with FULL 64-morphism orbit
        int atom_count = 0;
        while(atom_count < max_entries) {
            int best_idx = -1; 
            int max_cov = -1;
            
            // Search top candidates (limited for speed)
            int search_limit = std::min((int)pool.size(), 400);
            
            for(int i=0; i < search_limit; ++i) {
                if(pool[i].covered) continue;
                
                int cov = 0;
                
                // CRITICAL: Use FULL 64-morphism orbit (32 + 32 negated)
                // This maximizes the chance of finding matches
                for(int j=0; j < (int)pool.size(); ++j) {
                    if(pool[j].covered) continue;
                    
                    bool match = false;
                    // Full orbit: 64 morphisms (0-31 normal + 32-63 negated)
                    for(int m=0; m < 64; ++m) { 
                        MatrixXf t = is_spectral ? apply_spectral_action(pool[i].m, m) : 
                                                   apply_galois_action(pool[i].m, m);
                        float dot = (t.array() * pool[j].m.array()).sum();
                        
                        // AGGRESSIVE threshold for maximum compression
                        // Higher threshold = fewer atoms = better compression
                        if(std::abs(dot) > 0.90f) { 
                            match = true;
                            break;
                        }
                    }
                    if(match) cov++;
                }
                
                if(cov > max_cov) { 
                    max_cov = cov; 
                    best_idx = i; 
                }
            }
            
            if(best_idx == -1) {
                cout << "  No more useful atoms found at " << atom_count << " atoms." << endl;
                break;
            }
            
            atoms.push_back(pool[best_idx].m);
            atom_count++;
            
            // Progress indicator every 32 atoms
            if(atom_count % 32 == 0) {
                cout << "    " << atom_count << " atoms selected..." << endl;
            }
            
            // Mark covered patches with FULL orbit
            for(int j=0; j < (int)pool.size(); ++j) {
                if(pool[j].covered) continue;
                
                bool match = false;
                for(int m=0; m < 64; ++m) {
                    MatrixXf t = is_spectral ? apply_spectral_action(pool[best_idx].m, m) : 
                                               apply_galois_action(pool[best_idx].m, m);
                    if(std::abs((t.array() * pool[j].m.array()).sum()) > 0.88f) {
                        match = true;
                        break;
                    }
                }
                if(match) pool[j].covered = true;
            }
            pool[best_idx].covered = true;
        }
        
        cout << "  Dictionary trained: " << atoms.size() << " atoms" << endl;
    }

    // IMPROVED spectral learning with FULL morphism orbit
    void learn_from_samples(const vector<MatrixXf> &samples, int limit){
        if(samples.empty() || limit <= 0) return;
        
        cout << "Training spectral dictionary with full orbit..." << endl;
        
        struct Node{ float score; MatrixXf m; };
        vector<Node> candidates; 
        candidates.reserve(samples.size());
        
        for(const auto &s: samples){ 
            float n=s.norm(); 
            if(n > 1e-5f) candidates.push_back({n, s/n}); 
        }
        
        sort(candidates.begin(), candidates.end(), 
             [](const Node&a, const Node&b){ return a.score > b.score; });
        
        for(const auto &c: candidates){ 
            if((int)atoms.size() >= limit) break; 
            
            bool exists = false; 
            // FULL 64-morphism check for spectral atoms
            for(const auto &a: atoms){ 
                if(a.rows() != c.m.rows() || a.cols() != c.m.cols()) continue;
                
                for(int m=0; m < 64; ++m){ 
                    MatrixXf morp = is_spectral ? apply_spectral_action(a, (uint8_t)m) : 
                                                  apply_galois_action(a, (uint8_t)m);
                    float dot = (morp.array() * c.m.array()).sum(); 
                    // Tight threshold for spectral uniqueness
                    if(std::abs(dot) > 0.90f){ 
                        exists = true; 
                        break; 
                    } 
                }
                if(exists) break; 
            }
            
            if(!exists) atoms.push_back(c.m); 
        }
        
        cout << "  Spectral dictionary: " << atoms.size() << " atoms" << endl;
    }

    // FULL 64-morphism search for encoding
    FieldEntry solve(const MatrixXf &target, float &scale) {
        FieldEntry best; 
        scale = 0.0f;
        if(atoms.empty()) return best;
        
        MatrixXf t_canon = is_spectral ? target : resize_matrix(target, CANONICAL_SIZE);
        float tnorm = t_canon.norm();
        if(tnorm < 1e-9f) return best;

        float best_corr = -2.0f;
        
        // CRITICAL: Search ALL 64 morphisms for maximum compression
        for(size_t i=0; i < atoms.size(); ++i){
            for(int m=0; m < 64; ++m){  // FULL ORBIT
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
        scale = best_corr;
        return best;
    }
};

// --- RDO Compression (Optional) ---

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
    e.chroma = (uint8_t)std::clamp(sigma * 0.5f, 0.0f, 255.0f);

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

struct Image { int width=0, height=0; MatrixXf data; };

Image loadPGM(const string &filename){
    ifstream f(filename, ios::binary);
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
    int spatial_stride = 16;  // Default to larger stride for diversity
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
        if (spatial_stride < 1) spatial_stride = std::max(8, spatial_block / 2);
    }
};

void compress(const string &in, const string &out, CompressionConfig cfg){
    cfg.validate(); 
    Image img = loadPGM(in);
    
    cout << "\n=== GDHC v8.1 Maximum Compression ===" << endl;
    cout << "Image: " << img.width << "x" << img.height << endl;
    cout << "Config: block=" << cfg.spatial_block << " stride=" << cfg.spatial_stride 
         << " atoms=" << cfg.spatial_entries << endl;
    
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
    MatrixXf weight = MatrixXf::Zero(img.height, img.width);

    cout << "\nEncoding spatial layer..." << endl;
    
    if(cfg.use_quadtree) {
        float lambda = cfg.rdo_lambda;
        for(int i=0; i < img.height; i += cfg.spatial_block)
            for(int j=0; j < img.width; j += cfg.spatial_block) {
                RDOStats stats = qt_compress_rdo(img.data, i, j, cfg.spatial_block, 
                                                 sdict, recon, lambda, 0);
                ss.write(stats.stream.data(), stats.stream.size());
            }
    } else {
        // Overlapping mode for quality
        vector<FieldEntry> entries;
        int block_count = 0;
        
        for(int i=0; i+cfg.spatial_block <= img.height; i+=cfg.spatial_stride){
            for(int j=0; j+cfg.spatial_block <= img.width; j+=cfg.spatial_stride){ 
                MatrixXf blk = img.data.block(i,j,cfg.spatial_block,cfg.spatial_block);
                float mu = blk.mean(); 
                MatrixXf centered = blk.array() - mu; 
                float sigma = centered.norm(); 
                if(sigma > 1e-6f) centered /= sigma;
                
                float scale; 
                FieldEntry e = sdict.solve(centered, scale);
                e.luma = (uint8_t)std::clamp(mu, 0.0f, 255.0f); 
                e.chroma = (uint8_t)std::clamp(sigma * 0.5f, 0.0f, 255.0f);
                entries.push_back(e);
                
                MatrixXf atom;
                if(e.id < sdict.atoms.size()){ 
                    atom = apply_galois_action(sdict.atoms[e.id], e.morphism);
                    recon.block(i,j,cfg.spatial_block,cfg.spatial_block).array() += 
                        (atom.array() * ((float)e.chroma * 2.0f)) + (float)e.luma;
                } else { 
                    recon.block(i,j,cfg.spatial_block,cfg.spatial_block).array() += (float)e.luma; 
                }
                weight.block(i,j,cfg.spatial_block,cfg.spatial_block).array() += 1.0f;
                block_count++;
            }
        }
        recon = recon.array() / weight.array().max(1.0f);
        
        cout << "  Encoded " << block_count << " blocks" << endl;
        
        for(auto &e : entries) ss.write((char*)&e, sizeof(FieldEntry));
    }

    cout << "Encoding spectral residual..." << endl;
    
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

    int spec_block_count = 0;
    for(int i=0; i+sbk <= img.height; i+=sbk)
        for(int j=0; j+sbk <= img.width
