/**
 * Galois-DCT Hybrid Compressor (GDHC) - v10.0 "Tunable Crypto-Residual"
 * * ARCHITECTURE:
 * 1. Dual-Layer: Base Layer (Coarse) + Residual Layer (Fine/Texture).
 * 2. Dictionary Learning: Two separate dictionaries trained on-the-fly.
 * 3. Morphisms: Geometric (D4) + Cryptographic (Permutations).
 * 4. Tunable: All heuristic parameters exposed via CLI.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <sstream>
#include <cstdint>
#include <cfloat>
#include <random>
#include <map>
#include <iomanip>
#include <Eigen/Dense>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/zlib.hpp>

using namespace Eigen;
using namespace std;

// --- Constants (Fixed Architecture) ---
const int CANONICAL_SIZE = 8; // Feature space resolution

// --- Configuration Structure ---
struct CompressionConfig {
    // Structural
    int base_block_size = 16;
    int resid_block_size = 4;
    
    // Dictionary Quality
    int base_entries = 128;
    int resid_entries = 256;
    float base_var_threshold = 100.0f;  // Minimum detail to trigger atom learning
    float resid_var_threshold = 10.0f;
    
    // RDO / Quality
    float lambda_base = 150.0f;   // Higher = prefer bits over quality (smoother base)
    float lambda_resid = 20.0f;   // Lower = prefer quality (preserve texture)
    
    // Search Effort
    int crypto_trials = 64;       // Number of random permutations to test per block
};

enum QTNodeType : uint8_t { QT_LEAF = 0, QT_SPLIT = 1 };

#pragma pack(push,1)
struct FieldEntry { 
    uint16_t id = 0;       
    uint16_t morphism = 0; 
    uint8_t offset = 0;    
    uint8_t gain = 0;      
};
#pragma pack(pop)

struct RDOStats {
    float ssd = FLT_MAX;
    float bits = 0;
    std::string stream;
};

// --- Math Helpers ---

MatrixXf resize_matrix(const MatrixXf& src, int target_size) {
    if (src.rows() == target_size && src.cols() == target_size) return src;
    MatrixXf dst(target_size, target_size);
    float scale_r = (float)src.rows() / target_size;
    float scale_c = (float)src.cols() / target_size;

    for (int r = 0; r < target_size; ++r) {
        for (int c = 0; c < target_size; ++c) {
            if (scale_r >= 1.0f) { // Downscale
                int sr_start = (int)(r * scale_r);
                int sr_end = std::min((int)((r + 1) * scale_r), (int)src.rows());
                int sc_start = (int)(c * scale_c);
                int sc_end = std::min((int)((c + 1) * scale_c), (int)src.cols());
                if (sr_end <= sr_start) sr_end = sr_start + 1;
                if (sc_end <= sc_start) sc_end = sc_start + 1;
                float sum = src.block(sr_start, sc_start, sr_end - sr_start, sc_end - sc_start).sum();
                dst(r,c) = sum / ((sr_end - sr_start) * (sc_end - sc_start));
            } else { // Upscale
                int sr = std::min((int)(r * scale_r), (int)src.rows() - 1);
                int sc = std::min((int)(c * scale_c), (int)src.cols() - 1);
                dst(r,c) = src(sr, sc); 
            }
        }
    }
    return dst;
}

// --- Cryptographic Morphism Engine ---

class CryptoMorphismEngine {
public:
    static MatrixXf apply(const MatrixXf& src, uint16_t key) {
        if (key < 16) return apply_geometric(src, key);

        // Pseudo-Random Permutation (PRP) of coordinates
        int rows = src.rows();
        int cols = src.cols();
        int n = rows * cols;
        MatrixXf dst(rows, cols);

        // Standard mersenne_twister_engine seeded with the 'morphism key'
        // This ensures the decoder can replicate the shuffle exactly.
        std::mt19937 rng(key);
        std::vector<int> indices(n);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);

        const float* src_ptr = src.data(); 
        float* dst_ptr = dst.data();

        for(int i=0; i<n; ++i) {
            dst_ptr[i] = src_ptr[indices[i]];
        }
        
        return dst;
    }

private:
    static MatrixXf apply_geometric(const MatrixXf& block, uint16_t m) {
        MatrixXf res;
        bool negate = (m & 8);
        uint8_t op = m & 7;

        switch(op) {
            case 0: res = block; break;
            case 1: res = block.reverse(); break;
            case 2: res = block.rowwise().reverse(); break;
            case 3: res = block.colwise().reverse(); break;
            case 4: res = block.transpose(); break;
            case 5: res = block.transpose().reverse(); break;
            case 6: res = block.transpose().rowwise().reverse(); break;
            case 7: res = block.transpose().colwise().reverse(); break;
        }
        if (negate) res = -res;
        return res;
    }
};

// --- Dictionary ---

class ManifoldDictionary {
public:
    vector<MatrixXf> atoms;

    void train(const MatrixXf &data, int block_size, int max_entries, float min_var) {
        atoms.clear();
        struct Candidate { MatrixXf m; float var; };
        vector<Candidate> pool;

        int step = block_size / 2;
        // Collect high-variance patches
        for(int i=0; i <= data.rows()-block_size; i+=step) {
            for(int j=0; j <= data.cols()-block_size; j+=step) {
                MatrixXf b = data.block(i,j,block_size,block_size);
                MatrixXf b_centered = b.array() - b.mean();
                float v = b_centered.squaredNorm();
                if (v < min_var) continue;
                
                MatrixXf canon = resize_matrix(b_centered, CANONICAL_SIZE);
                float n = canon.norm();
                if (n > 1e-5) pool.push_back({canon/n, v});
            }
        }

        if (pool.empty()) return;

        // Sort by energy
        std::sort(pool.begin(), pool.end(), [](auto& a, auto& b){ return a.var > b.var; });
        
        // Greedy selection (Farthest Point Sampling approximation)
        if(!pool.empty()) atoms.push_back(pool[0].m);
        
        int samples = std::min((int)pool.size(), max_entries * 15);
        while((int)atoms.size() < max_entries && atoms.size() < pool.size()) {
            int best_idx = -1;
            float max_dist = -1;
            
            for(int i=0; i<samples; ++i) {
                float min_d = FLT_MAX;
                // Only check against last few added atoms for speed
                int check_start = std::max(0, (int)atoms.size() - 20);
                for(int k=check_start; k<(int)atoms.size(); ++k) 
                    min_d = std::min(min_d, (pool[i].m - atoms[k]).norm());
                
                if (min_d > max_dist) { max_dist = min_d; best_idx = i; }
            }
            if (best_idx != -1) atoms.push_back(pool[best_idx].m);
            else break;
        }
    }

    FieldEntry solve(const MatrixXf &target, float &scale, int max_trials) {
        FieldEntry best; scale = 0.0f;
        if(atoms.empty()) return best;

        MatrixXf t_canon = resize_matrix(target, CANONICAL_SIZE);
        float tnorm = t_canon.norm();
        if(tnorm < 1e-9f) return best;
        t_canon /= tnorm; 

        float best_corr = -1.0f;

        // 1. Check Standard Geometric Morphisms
        for(size_t i=0; i < atoms.size(); ++i){
            for(uint16_t m=0; m < 16; ++m){
                MatrixXf cand = CryptoMorphismEngine::apply(atoms[i], m);
                float corr = (cand.array() * t_canon.array()).sum();
                if(corr > best_corr){
                    best_corr = corr;
                    best.id = (uint16_t)i;
                    best.morphism = m;
                }
            }
        }

        // 2. Check "Crypto" Morphisms (Monte Carlo Search)
        // If the match is poor, try scrambling the best candidates
        if (best_corr < 0.96f && max_trials > 0) {
            std::mt19937 rng(12345); 
            std::uniform_int_distribution<uint16_t> dist_key(16, 65535);
            
            size_t best_atom_idx = best.id; // Scramble the geometrically best atom
            
            for (int trial=0; trial < max_trials; ++trial) {
                uint16_t key = dist_key(rng);
                MatrixXf cand = CryptoMorphismEngine::apply(atoms[best_atom_idx], key);
                float corr = (cand.array() * t_canon.array()).sum();
                
                if (corr > best_corr) {
                    best_corr = corr;
                    best.id = (uint16_t)best_atom_idx;
                    best.morphism = key;
                }
            }
        }

        scale = best_corr * tnorm; 
        return best;
    }
};

// --- Image & Color ---

struct Image3 { 
    int width=0, height=0; 
    std::vector<MatrixXf> channels; // Y, Co, Cg
    Image3() { channels.resize(3); }
};

void RGB_to_YCoCg(const std::vector<unsigned char>& rgb, Image3& img) {
    int w = img.width, h = img.height;
    for(int k=0; k<3; ++k) img.channels[k].resize(h, w);
    for(int i=0; i<h; ++i) {
        for(int j=0; j<w; ++j) {
            int idx = (i*w + j) * 3;
            float r = rgb[idx], g = rgb[idx+1], b = rgb[idx+2];
            img.channels[0](i,j) = r/4.0f + g/2.0f + b/4.0f; 
            img.channels[1](i,j) = r/2.0f - b/2.0f;          
            img.channels[2](i,j) = -r/4.0f + g/2.0f - b/4.0f; 
        }
    }
}

void YCoCg_to_RGB(const Image3& img, std::vector<unsigned char>& rgb) {
    int w = img.width, h = img.height;
    rgb.resize(w*h*3);
    for(int i=0; i<h; ++i) {
        for(int j=0; j<w; ++j) {
            float y = img.channels[0](i,j), co = img.channels[1](i,j), cg = img.channels[2](i,j);
            float tmp = y - cg;
            float r = tmp + co;
            float g = y + cg;
            float b = tmp - co;
            int idx = (i*w + j) * 3;
            rgb[idx]   = (unsigned char)std::clamp(r, 0.0f, 255.0f);
            rgb[idx+1] = (unsigned char)std::clamp(g, 0.0f, 255.0f);
            rgb[idx+2] = (unsigned char)std::clamp(b, 0.0f, 255.0f);
        }
    }
}

Image3 loadPPM(const string &filename){
    ifstream f(filename, ios::binary);
    if (!f) { throw std::runtime_error("Cannot open input file"); }
    string magic; f >> magic;
    auto skip=[&](ifstream &in){ while(isspace(in.peek())) in.get(); while(in.peek()=='#'){ string d; getline(in,d);} };
    skip(f); int w, h, maxv; f >> w; skip(f); f >> h; skip(f); f >> maxv; f.get();
    Image3 img; img.width=w; img.height=h;
    vector<unsigned char> buf((size_t)w*h*3);
    f.read((char*)buf.data(), buf.size());
    RGB_to_YCoCg(buf, img);
    return img;
}

void savePPM(const string &filename, const Image3 &img){
    ofstream f(filename, ios::binary);
    f<<"P6\n"<<img.width<<" "<<img.height<<"\n255\n";
    vector<unsigned char> buf;
    YCoCg_to_RGB(img, buf);
    f.write((char*)buf.data(), buf.size());
}

// --- CORE COMPRESSOR LOGIC ---

RDOStats compress_block_rdo(const MatrixXf &src, int r, int c, int size, 
                           ManifoldDictionary &dict, MatrixXf &recon_out, 
                           float lambda, int crypto_trials, bool allow_split, int min_split_size) {
    RDOStats res;
    int h_eff = std::min(size, (int)src.rows() - r);
    int w_eff = std::min(size, (int)src.cols() - c);
    
    MatrixXf blk = src.block(r, c, h_eff, w_eff);
    float mu = blk.mean();
    MatrixXf centered = blk.array() - mu;
    
    // 1. Solve using Dictionary (Base or Residual)
    float scale_found = 0;
    FieldEntry e = dict.solve(centered, scale_found, crypto_trials);

    e.offset = (uint8_t)std::clamp(mu + 128.0f, 0.0f, 255.0f);
    
    // Adaptive quantization
    float q_step = (allow_split) ? 1.0f : 0.5f; 
    float gain_val = std::abs(scale_found);
    e.gain = (uint8_t)std::clamp(gain_val / q_step, 0.0f, 255.0f);

    // Reconstruct
    MatrixXf atom;
    if (!dict.atoms.empty() && e.id < dict.atoms.size()) {
        atom = resize_matrix(CryptoMorphismEngine::apply(dict.atoms[e.id], e.morphism), size);
        if (scale_found < 0 && e.morphism >= 16) atom = -atom; 
        
        float n = atom.norm();
        if (n > 1e-6f) atom *= ((float)e.gain * q_step / n);
        if (scale_found < 0 && e.morphism < 16) atom = -atom; 
    } else {
        atom = MatrixXf::Zero(size, size);
    }
    
    MatrixXf leaf_recon = (atom.array() + (float)(e.offset - 128)).block(0,0,h_eff,w_eff);

    float ssd = (blk - leaf_recon).squaredNorm();
    // Bit cost estimation (approximate): ID(8)+Morph(16)+Off(8)+Gain(8) + overhead
    float bits = 40.0f; 
    float cost = ssd + lambda * bits;

    // Split Logic (Quadtree)
    if (allow_split && size > min_split_size) {
        int half = size / 2;
        RDOStats children[4];
        float split_ssd = 0, split_bits = 4.0f; // Split flag bits
        
        children[0] = compress_block_rdo(src, r, c, half, dict, recon_out, lambda, crypto_trials, true, min_split_size);
        children[1] = compress_block_rdo(src, r, c+half, half, dict, recon_out, lambda, crypto_trials, true, min_split_size);
        children[2] = compress_block_rdo(src, r+half, c, half, dict, recon_out, lambda, crypto_trials, true, min_split_size);
        children[3] = compress_block_rdo(src, r+half, c+half, half, dict, recon_out, lambda, crypto_trials, true, min_split_size);

        for(int i=0; i<4; ++i) { split_ssd += children[i].ssd; split_bits += children[i].bits; }
        
        if (split_ssd + lambda * split_bits < cost) {
            res.ssd = split_ssd; res.bits = split_bits;
            uint8_t flag = QT_SPLIT;
            res.stream.append((char*)&flag, 1);
            for(int i=0; i<4; ++i) res.stream.append(children[i].stream);
            return res;
        }
    }

    // Leaf Decision
    recon_out.block(r, c, h_eff, w_eff) = leaf_recon;
    res.ssd = ssd; res.bits = bits;
    if (allow_split) { uint8_t flag = QT_LEAF; res.stream.append((char*)&flag, 1); }
    res.stream.append((char*)&e, sizeof(FieldEntry));
    return res;
}

void compress(const string &in, const string &out, const CompressionConfig& cfg) {
    Image3 img = loadPPM(in);
    stringstream ss;
    
    // Header
    int w = img.width, h = img.height;
    ss.write((char*)&w, 4); ss.write((char*)&h, 4);
    ss.write((char*)&cfg.base_block_size, 4);
    ss.write((char*)&cfg.resid_block_size, 4);

    // Buffers
    Image3 recon_base; recon_base.width=w; recon_base.height=h;
    for(int k=0; k<3; ++k) recon_base.channels[k] = MatrixXf::Zero(h,w);

    // --- PASS 1: Base Layer (Structure) ---
    cout << "Training Base Dictionary (" << cfg.base_entries << " atoms, " << cfg.base_block_size << "x" << cfg.base_block_size << ")..." << endl;
    ManifoldDictionary dict_base;
    dict_base.train(img.channels[0], cfg.base_block_size, cfg.base_entries, cfg.base_var_threshold); 
    
    uint32_t c = dict_base.atoms.size(); ss.write((char*)&c, 4);
    for(auto &a : dict_base.atoms) ss.write((char*)a.data(), a.size()*4);

    cout << "Compressing Base Layer..." << endl;
    for(int k=0; k<3; ++k) {
        for(int i=0; i<h; i+=cfg.base_block_size) {
            for(int j=0; j<w; j+=cfg.base_block_size) {
                // Base layer allows splitting down to half size
                RDOStats s = compress_block_rdo(img.channels[k], i, j, cfg.base_block_size, 
                                              dict_base, recon_base.channels[k], 
                                              cfg.lambda_base, cfg.crypto_trials, 
                                              true, cfg.base_block_size/2);
                ss.write(s.stream.data(), s.stream.size());
            }
        }
    }

    // --- PASS 2: Residual Layer (Details) ---
    Image3 resid; resid.width=w; resid.height=h;
    for(int k=0; k<3; ++k) resid.channels[k] = img.channels[k] - recon_base.channels[k];

    cout << "Training Residual Dictionary (" << cfg.resid_entries << " atoms, " << cfg.resid_block_size << "x" << cfg.resid_block_size << ")..." << endl;
    ManifoldDictionary dict_resid;
    dict_resid.train(resid.channels[0], cfg.resid_block_size, cfg.resid_entries, cfg.resid_var_threshold); 

    c = dict_resid.atoms.size(); ss.write((char*)&c, 4);
    for(auto &a : dict_resid.atoms) ss.write((char*)a.data(), a.size()*4);

    cout << "Compressing Residuals..." << endl;
    for(int k=0; k<3; ++k) {
        for(int i=0; i<h; i+=cfg.resid_block_size) {
            for(int j=0; j<w; j+=cfg.resid_block_size) {
                // Residual layer usually fixed block size for speed/density
                RDOStats s = compress_block_rdo(resid.channels[k], i, j, cfg.resid_block_size, 
                                              dict_resid, recon_base.channels[k], 
                                              cfg.lambda_resid, cfg.crypto_trials, 
                                              false, 0);
                ss.write(s.stream.data(), s.stream.size()); 
            }
        }
    }

    ofstream ofs(out, ios::binary);
    boost::iostreams::filtering_streambuf<boost::iostreams::output> fb;
    fb.push(boost::iostreams::zlib_compressor());
    fb.push(ofs);
    boost::iostreams::copy(ss, fb);
    cout << "Compression Complete." << endl;
}

// --- DECODER ---

void decode_block(stringstream &ss, MatrixXf &target, int r, int c, int size, 
                  const vector<MatrixXf> &atoms, bool allow_split) {
    if (allow_split) {
        uint8_t flag; ss.read((char*)&flag, 1);
        if (flag == QT_SPLIT) {
            int half = size/2;
            decode_block(ss, target, r, c, half, atoms, true);
            decode_block(ss, target, r, c+half, half, atoms, true);
            decode_block(ss, target, r+half, c, half, atoms, true);
            decode_block(ss, target, r+half, c+half, half, atoms, true);
            return;
        }
    }

    FieldEntry e; ss.read((char*)&e, sizeof(FieldEntry));
    int h_eff = std::min(size, (int)target.rows()-r);
    int w_eff = std::min(size, (int)target.cols()-c);
    
    MatrixXf atom;
    if (e.id < atoms.size()) {
        atom = resize_matrix(CryptoMorphismEngine::apply(atoms[e.id], e.morphism), size);
        
        float q_step = allow_split ? 1.0f : 0.5f;
        float gain = (float)e.gain * q_step;
        
        float n = atom.norm();
        if (n > 1e-6f) atom *= (gain / n);
    } else {
        atom = MatrixXf::Zero(size, size);
    }

    // Fix: Explicitly convert Array to Matrix for the addition to avoid the mixed-type error.
    target.block(r, c, h_eff, w_eff) += (atom.array() + (float)(e.offset - 128)).block(0, 0, h_eff, w_eff).matrix();
}

void decompress(const string &in, const string &out) {
    ifstream ifs(in, ios::binary); 
    if (!ifs) { cerr << "Error opening input file\n"; return; }
    
    stringstream ss;
    boost::iostreams::filtering_streambuf<boost::iostreams::input> fb;
    fb.push(boost::iostreams::zlib_decompressor());
    fb.push(ifs);
    boost::iostreams::copy(fb, ss);

    int w, h, base_bs, resid_bs;
    ss.read((char*)&w, 4); ss.read((char*)&h, 4);
    ss.read((char*)&base_bs, 4); ss.read((char*)&resid_bs, 4);
    
    Image3 img; img.width=w; img.height=h;
    for(int k=0; k<3; ++k) img.channels[k] = MatrixXf::Zero(h,w);

    // 1. Read Base Dictionary
    uint32_t c; ss.read((char*)&c, 4);
    vector<MatrixXf> atoms_base(c, MatrixXf(CANONICAL_SIZE, CANONICAL_SIZE));
    for(uint32_t i=0; i<c; ++i) ss.read((char*)atoms_base[i].data(), CANONICAL_SIZE*CANONICAL_SIZE*4);

    // 2. Decode Base Layer
    for(int k=0; k<3; ++k)
        for(int i=0; i<h; i+=base_bs)
            for(int j=0; j<w; j+=base_bs)
                decode_block(ss, img.channels[k], i, j, base_bs, atoms_base, true);

    // 3. Read Residual Dictionary
    ss.read((char*)&c, 4);
    vector<MatrixXf> atoms_resid(c, MatrixXf(CANONICAL_SIZE, CANONICAL_SIZE));
    for(uint32_t i=0; i<c; ++i) ss.read((char*)atoms_resid[i].data(), CANONICAL_SIZE*CANONICAL_SIZE*4);

    // 4. Decode Residual Layer
    for(int k=0; k<3; ++k)
        for(int i=0; i<h; i+=resid_bs)
            for(int j=0; j<w; j+=resid_bs)
                decode_block(ss, img.channels[k], i, j, resid_bs, atoms_resid, false);

    savePPM(out, img);
}

// --- ARGUMENT PARSER ---

void print_help(const char* prog) {
    cout << "Usage: " << prog << " <mode:c/d> <input.ppm/bin> <output> [options]\n"
         << "Options:\n"
         << "  --base-block <int>    Size of base layer blocks (default: 16)\n"
         << "  --resid-block <int>   Size of residual micro-blocks (default: 4)\n"
         << "  --base-entries <int>  Dictionary size for base layer (default: 128)\n"
         << "  --resid-entries <int> Dictionary size for residual layer (default: 256)\n"
         << "  --lambda-base <float> RDO Lambda for base (higher=less bits) (default: 150.0)\n"
         << "  --lambda-resid <float> RDO Lambda for residual (default: 20.0)\n"
         << "  --base-var <float>    Min variance to learn base atom (default: 100.0)\n"
         << "  --resid-var <float>   Min variance to learn residual atom (default: 10.0)\n"
         << "  --base-entries <int>  Dictionary size for base layer (default: 128)\n"
         << "  --resid-entries <int> Dictionary size for residual layer (default: 256)\n"
         << "  --crypto-trials <int> Random permutation trials per block (default: 64)\n";
}

int main(int argc, char** argv) {
    if(argc < 4) { print_help(argv[0]); return 1; }

    string mode = argv[1];
    string in_file = argv[2];
    string out_file = argv[3];

    CompressionConfig cfg;

    // Parse tunable parameters
    for(int i=4; i<argc; ++i) {
        string arg = argv[i];
        if(i+1 >= argc) break; 
        
        try {
            if(arg == "--base-block") cfg.base_block_size = stoi(argv[++i]);
            else if(arg == "--resid-block") cfg.resid_block_size = stoi(argv[++i]);
            else if(arg == "--base-entries") cfg.base_entries = stoi(argv[++i]);
            else if(arg == "--resid-entries") cfg.resid_entries = stoi(argv[++i]);
            else if(arg == "--lambda-base") cfg.lambda_base = stof(argv[++i]);
            else if(arg == "--lambda-resid") cfg.lambda_resid = stof(argv[++i]);
            else if(arg == "--base-var") cfg.base_var_threshold = stof(argv[++i]);
            else if(arg == "--resid-var") cfg.resid_var_threshold = stof(argv[++i]);
            else if(arg == "--crypto-trials") cfg.crypto_trials = stoi(argv[++i]);
        } catch(...) {
            cerr << "Invalid value for argument: " << arg << endl;
            return 1;
        }
    }

    if(mode == "c") compress(in_file, out_file, cfg);
    else if(mode == "d") decompress(in_file, out_file); 
    else { cerr << "Unknown mode: " << mode << endl; return 1; }

    return 0;
}
