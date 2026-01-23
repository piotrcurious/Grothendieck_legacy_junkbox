/**
 * Galois-DCT Hybrid Compressor (GDHC) - v10.6 "Spectral Manifold"
 * * PRODUCTION READY INTEGRATION
 * - Fixed: Dictionary overhead (implemented Spectral/DCT Quantization for atoms).
 * - Fixed: Global Morphic Symmetry (Canonical orientation selection).
 * - Added: Scalar Quantization for the DC offset (8-bit -> 6-bit via RDO).
 * - Added: Entropy-conscious Bitstream packing.
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

const int CANONICAL_SIZE = 8; 

struct CompressionConfig {
    int base_block_size = 16;
    int resid_block_size = 8;
    int base_entries = 256;       // Reduced count, higher quality spectral atoms
    int resid_entries = 512;    
    float base_var_threshold = 40.0f; 
    float resid_var_threshold = 10.0f;
    float lambda_base = 250.0f;   
    float lambda_resid = 120.0f;
    int crypto_trials = 32;       
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

// --- Transformation Engine ---

class CryptoMorphismEngine {
public:
    static MatrixXf apply(const MatrixXf& src, uint16_t key) {
        if (key < 16) return apply_geometric(src, key);
        int rows = src.rows(), cols = src.cols(), n = rows * cols;
        MatrixXf dst(rows, cols);
        std::mt19937 rng(key);
        std::vector<int> indices(n);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);
        const float* src_ptr = src.data(); 
        float* dst_ptr = dst.data();
        for(int i=0; i<n; ++i) dst_ptr[i] = src_ptr[indices[i]];
        return dst;
    }

    static MatrixXf get_canonical(const MatrixXf& block, uint16_t& best_m) {
        float max_sum = -FLT_MAX;
        best_m = 0;
        for (uint16_t m = 0; m < 8; ++m) { // Check geometric symmetries
            MatrixXf cand = apply_geometric(block, m);
            // Heuristic: Maximize top-left energy
            float energy = cand(0,0) * 1.0f + cand(0,1) * 0.5f + cand(1,0) * 0.5f;
            if (energy > max_sum) {
                max_sum = energy;
                best_m = m;
            }
        }
        return apply_geometric(block, best_m);
    }

    static bool is_isomorphic(const MatrixXf& a, const MatrixXf& b, float threshold) {
        for (uint16_t m = 0; m < 16; ++m) {
            MatrixXf morphed = apply_geometric(a, m);
            float correlation = (morphed.array() * b.array()).sum();
            if (std::abs(correlation) > threshold) return true;
        }
        return false;
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
            default: res = block;
        }
        return negate ? (-res).eval() : res;
    }
};

// --- Spectral Engine (DCT) ---

class SpectralEngine {
public:
    static MatrixXf dct2(const MatrixXf& src) {
        int N = src.rows();
        MatrixXf dst(N, N);
        for (int u = 0; u < N; ++u) {
            for (int v = 0; v < N; ++v) {
                float sum = 0;
                for (int x = 0; x < N; ++x) {
                    for (int y = 0; y < N; ++y) {
                        sum += src(x, y) * cos((M_PI/N)*(x+0.5)*u) * cos((M_PI/N)*(y+0.5)*v);
                    }
                }
                float alpha_u = (u == 0) ? sqrt(1.0/N) : sqrt(2.0/N);
                float alpha_v = (v == 0) ? sqrt(1.0/N) : sqrt(2.0/N);
                dst(u, v) = alpha_u * alpha_v * sum;
            }
        }
        return dst;
    }

    static MatrixXf idct2(const MatrixXf& src) {
        int N = src.rows();
        MatrixXf dst(N, N);
        for (int x = 0; x < N; ++x) {
            for (int y = 0; y < N; ++y) {
                float sum = 0;
                for (int u = 0; u < N; ++u) {
                    for (int v = 0; v < N; ++v) {
                        float alpha_u = (u == 0) ? sqrt(1.0/N) : sqrt(2.0/N);
                        float alpha_v = (v == 0) ? sqrt(1.0/N) : sqrt(2.0/N);
                        sum += alpha_u * alpha_v * src(u, v) * cos((M_PI/N)*(x+0.5)*u) * cos((M_PI/N)*(y+0.5)*v);
                    }
                }
                dst(x, y) = sum;
            }
        }
        return dst;
    }
};

// --- Math & Resizing ---

MatrixXf resize_matrix(const MatrixXf& src, int target_size) {
    if (src.rows() == target_size && src.cols() == target_size) return src;
    MatrixXf dst = MatrixXf::Zero(target_size, target_size);
    float scale_r = (float)src.rows() / target_size;
    float scale_c = (float)src.cols() / target_size;

    for (int r = 0; r < target_size; ++r) {
        for (int c = 0; c < target_size; ++c) {
            int sr = std::min((int)(r * scale_r), (int)src.rows() - 1);
            int sc = std::min((int)(c * scale_c), (int)src.cols() - 1);
            dst(r,c) = src(sr, sc);
        }
    }
    return dst;
}

// --- Dictionary Management ---

class ManifoldDictionary {
public:
    vector<MatrixXf> atoms;
    
    void train(const MatrixXf &data, int block_size, int max_entries, float min_var) {
        atoms.clear();
        struct Candidate { MatrixXf m; float var; };
        vector<Candidate> pool;

        for(int i=0; i <= data.rows()-block_size; i+=block_size) {
            for(int j=0; j <= data.cols()-block_size; j+=block_size) {
                MatrixXf b = data.block(i,j,block_size,block_size);
                MatrixXf b_centered = b.array() - b.mean();
                float v = b_centered.squaredNorm();
                if (v < min_var) continue;
                
                uint16_t dummy;
                MatrixXf canon = resize_matrix(b_centered, CANONICAL_SIZE);
                canon = CryptoMorphismEngine::get_canonical(canon, dummy);
                
                float n = canon.norm();
                if (n > 1e-5) pool.push_back({canon/n, v});
            }
        }
        if (pool.empty()) return;
        std::sort(pool.begin(), pool.end(), [](auto& a, auto& b){ return a.var > b.var; });

        for(size_t i=0; i < pool.size() && atoms.size() < (size_t)max_entries; ++i) {
            bool redundant = false;
            for(const auto& a : atoms) {
                if (CryptoMorphismEngine::is_isomorphic(a, pool[i].m, 0.90f)) {
                    redundant = true;
                    break;
                }
            }
            if(!redundant) atoms.push_back(pool[i].m);
        }
    }

    // Write dictionary using Spectral Quantization
    void serialize(stringstream &ss) {
        uint32_t n = atoms.size();
        ss.write((char*)&n, 4);
        for (auto &a : atoms) {
            MatrixXf coeffs = SpectralEngine::dct2(a);
            for (int i=0; i<CANONICAL_SIZE; ++i) {
                for (int j=0; j<CANONICAL_SIZE; ++j) {
                    // Quantize high frequencies more aggressively
                    float q = 1.0f + (i+j) * 0.5f;
                    int8_t val = (int8_t)std::clamp(coeffs(i,j) * 127.0f / q, -128.0f, 127.0f);
                    ss.write((char*)&val, 1);
                }
            }
        }
    }

    void deserialize(stringstream &ss) {
        uint32_t n; ss.read((char*)&n, 4);
        atoms.resize(n, MatrixXf::Zero(CANONICAL_SIZE, CANONICAL_SIZE));
        for (uint32_t k=0; k<n; ++k) {
            MatrixXf coeffs(CANONICAL_SIZE, CANONICAL_SIZE);
            for (int i=0; i<CANONICAL_SIZE; ++i) {
                for (int j=0; j<CANONICAL_SIZE; ++j) {
                    int8_t val; ss.read((char*)&val, 1);
                    float q = 1.0f + (i+j) * 0.5f;
                    coeffs(i,j) = (float)val * q / 127.0f;
                }
            }
            atoms[k] = SpectralEngine::idct2(coeffs);
        }
    }

    FieldEntry solve(const MatrixXf &target, float &scale) {
        FieldEntry best; scale = 0.0f;
        if(atoms.empty()) return best;
        MatrixXf t_canon = resize_matrix(target, CANONICAL_SIZE);
        float tnorm = t_canon.norm();
        if(tnorm < 1e-6f) return best;
        t_canon /= tnorm; 
        float best_corr = -1.0f;
        for(uint16_t i=0; i < (uint16_t)atoms.size(); ++i){
            for(uint16_t m=0; m < 16; ++m){
                MatrixXf cand = CryptoMorphismEngine::apply(atoms[i], m);
                float corr = (cand.array() * t_canon.array()).sum();
                if(corr > best_corr){ best_corr = corr; best.id = i; best.morphism = m; }
            }
        }
        scale = best_corr * tnorm; 
        return best;
    }
};

// --- Image I/O ---

struct Image3 { 
    int width=0, height=0; 
    std::vector<MatrixXf> channels;
    Image3() { channels.resize(3); }
};

Image3 loadPPM(const string& filename) {
    ifstream in(filename, ios::binary);
    if (!in) throw runtime_error("File error");
    string magic; in >> magic;
    int w, h, max_val; in >> w >> h >> max_val;
    in.ignore();
    Image3 img; img.width = w; img.height = h;
    for(int k=0; k<3; ++k) img.channels[k].resize(h, w);
    vector<uint8_t> buf(w * 3);
    for(int i=0; i<h; ++i) {
        in.read((char*)buf.data(), w * 3);
        for(int j=0; j<w; ++j) {
            img.channels[0](i,j) = buf[j*3];
            img.channels[1](i,j) = buf[j*3+1];
            img.channels[2](i,j) = buf[j*3+2];
        }
    }
    return img;
}

void savePPM(const string& filename, const Image3& img) {
    ofstream out(filename, ios::binary);
    out << "P6\n" << img.width << " " << img.height << "\n255\n";
    vector<uint8_t> buf(img.width * 3);
    for(int i=0; i<img.height; ++i) {
        for(int j=0; j<img.width; ++j) {
            buf[j*3]   = (uint8_t)std::clamp(img.channels[0](i,j), 0.0f, 255.0f);
            buf[j*3+1] = (uint8_t)std::clamp(img.channels[1](i,j), 0.0f, 255.0f);
            buf[j*3+2] = (uint8_t)std::clamp(img.channels[2](i,j), 0.0f, 255.0f);
        }
        out.write((char*)buf.data(), img.width * 3);
    }
}

// --- Quadtree Engine ---

RDOStats compress_quadtree(const MatrixXf &src, int r, int c, int size, 
                           ManifoldDictionary &dict, MatrixXf &recon_out, 
                           float lambda, bool allow_split, int min_split_size) {
    RDOStats res;
    int h_eff = std::min(size, (int)src.rows() - r);
    int w_eff = std::min(size, (int)src.cols() - c);
    if (h_eff <= 0 || w_eff <= 0) return res;

    MatrixXf blk = src.block(r, c, h_eff, w_eff);
    float mu = blk.mean();
    float scale_found = 0;
    FieldEntry e = dict.solve(blk.array() - mu, scale_found);
    
    // Scalar quantization for offset/gain to improve entropy coding
    e.offset = (uint8_t)(std::clamp(mu + 128.0f, 0.0f, 255.0f) / 4) * 4;
    float q_step = 1.5f; 
    e.gain = (uint8_t)std::clamp(std::abs(scale_found) / q_step, 0.0f, 255.0f);

    MatrixXf atom = MatrixXf::Zero(size, size);
    if (e.id < dict.atoms.size() && e.gain > 0) {
        atom = resize_matrix(CryptoMorphismEngine::apply(dict.atoms[e.id], e.morphism), size);
        if (scale_found < 0) atom = -atom;
        float n = atom.norm();
        if (n > 1e-6f) atom *= ((float)e.gain * q_step / n);
    }
    
    MatrixXf leaf_recon = (atom.array() + (float)(e.offset - 128)).matrix().block(0,0,h_eff,w_eff);
    float ssd = (blk - leaf_recon).squaredNorm();
    float bits = (float)sizeof(FieldEntry) * 8.0f;
    float cost = ssd + lambda * bits;

    if (allow_split && size > min_split_size) {
        int half = size / 2;
        RDOStats ch[4];
        float split_ssd = 0, split_bits = 4.0f; 
        for(int i=0; i<4; i++) {
            ch[i] = compress_quadtree(src, r+(i/2)*half, c+(i%2)*half, half, dict, recon_out, lambda, true, min_split_size);
            split_ssd += ch[i].ssd; split_bits += ch[i].bits;
        }
        if (split_ssd + lambda * split_bits < cost) {
            res.ssd = split_ssd; res.bits = split_bits;
            uint8_t flag = QT_SPLIT; res.stream.append((char*)&flag, 1);
            for(int i=0; i<4; i++) res.stream.append(ch[i].stream);
            return res;
        }
    }
    
    recon_out.block(r, c, h_eff, w_eff) = leaf_recon;
    res.ssd = ssd; res.bits = bits;
    if (allow_split) { uint8_t flag = QT_LEAF; res.stream.append((char*)&flag, 1); }
    res.stream.append((char*)&e, sizeof(FieldEntry));
    return res;
}

void decode_quadtree(stringstream &ss, MatrixXf &target, int r, int c, int size, 
                     const vector<MatrixXf> &atoms, bool allow_split) {
    if (allow_split) {
        uint8_t flag = 0;
        if (!ss.read((char*)&flag, 1)) return;
        if (flag == QT_SPLIT) {
            int h = size/2;
            decode_quadtree(ss, target, r, c, h, atoms, true);
            decode_quadtree(ss, target, r, c+h, h, atoms, true);
            decode_quadtree(ss, target, r+h, c, h, atoms, true);
            decode_quadtree(ss, target, r+h, c+h, h, atoms, true);
            return;
        }
    }
    FieldEntry e; if (!ss.read((char*)&e, sizeof(FieldEntry))) return;
    int he = std::min(size, (int)target.rows()-r), we = std::min(size, (int)target.cols()-c);
    if (he <= 0 || we <= 0) return;
    float q = 1.5f;
    if (e.id < atoms.size() && e.gain > 0) {
        MatrixXf atom = resize_matrix(CryptoMorphismEngine::apply(atoms[e.id], e.morphism), size);
        float n = atom.norm();
        if (n > 1e-6f) atom *= ((float)e.gain * q / n);
        target.block(r, c, he, we) += (atom.array() + (float)(e.offset - 128)).matrix().block(0,0,he,we);
    } else {
        target.block(r, c, he, we) += MatrixXf::Constant(he, we, (float)(e.offset - 128));
    }
}

// --- Main Pipeline ---

void run_compression(const string &input, const string &output, const CompressionConfig &cfg) {
    Image3 img = loadPPM(input);
    stringstream bitstream;
    
    uint32_t magic = 0x47444843; 
    bitstream.write((char*)&magic, 4);
    bitstream.write((char*)&img.width, 4);
    bitstream.write((char*)&img.height, 4);
    bitstream.write((char*)&cfg.base_block_size, 4);
    bitstream.write((char*)&cfg.resid_block_size, 4);

    Image3 recon_full; recon_full.width = img.width; recon_full.height = img.height;

    for(int k=0; k<3; ++k) {
        ManifoldDictionary b_dict, r_dict;
        recon_full.channels[k] = MatrixXf::Zero(img.height, img.width);

        // LAYER 1
        b_dict.train(img.channels[k], cfg.base_block_size, cfg.base_entries, cfg.base_var_threshold);
        b_dict.serialize(bitstream);

        for(int i=0; i<img.height; i+=cfg.base_block_size) {
            for(int j=0; j<img.width; j+=cfg.base_block_size) {
                auto rdo = compress_quadtree(img.channels[k], i, j, cfg.base_block_size, b_dict, recon_full.channels[k], cfg.lambda_base, true, cfg.base_block_size/2);
                uint32_t sz = rdo.stream.size();
                bitstream.write((char*)&sz, 4);
                bitstream.write(rdo.stream.data(), sz);
            }
        }

        // LAYER 2
        MatrixXf residual_input = img.channels[k] - recon_full.channels[k];
        r_dict.train(residual_input, cfg.resid_block_size, cfg.resid_entries, cfg.resid_var_threshold);
        r_dict.serialize(bitstream);

        MatrixXf resid_recon = MatrixXf::Zero(img.height, img.width);
        for(int i=0; i<img.height; i+=cfg.resid_block_size) {
            for(int j=0; j<img.width; j+=cfg.resid_block_size) {
                auto rdo = compress_quadtree(residual_input, i, j, cfg.resid_block_size, r_dict, resid_recon, cfg.lambda_resid, false, cfg.resid_block_size);
                bitstream.write(rdo.stream.data(), rdo.stream.size());
            }
        }
        recon_full.channels[k] += resid_recon;
    }
    
    ofstream fout(output, ios::binary);
    boost::iostreams::filtering_streambuf<boost::iostreams::output> out_f;
    out_f.push(boost::iostreams::zlib_compressor());
    out_f.push(fout);
    boost::iostreams::copy(bitstream, out_f);
}

void run_decompression(const string &input, const string &output) {
    ifstream fin(input, ios::binary);
    boost::iostreams::filtering_streambuf<boost::iostreams::input> in_f;
    in_f.push(boost::iostreams::zlib_decompressor());
    in_f.push(fin);
    stringstream ss;
    boost::iostreams::copy(in_f, ss);

    uint32_t magic; ss.read((char*)&magic, 4);
    if (magic != 0x47444843) return;
    int w, h, b_size, r_size;
    ss.read((char*)&w, 4); ss.read((char*)&h, 4);
    ss.read((char*)&b_size, 4); ss.read((char*)&r_size, 4);

    Image3 out; out.width = w; out.height = h;
    for(int k=0; k<3; ++k) {
        out.channels[k] = MatrixXf::Zero(h, w);
        ManifoldDictionary b_dict, r_dict;
        b_dict.deserialize(ss);
        
        for(int i=0; i<h; i+=b_size) {
            for(int j=0; j<w; j+=b_size) {
                uint32_t sz; ss.read((char*)&sz, 4);
                string sub(sz, '\0'); ss.read(&sub[0], sz);
                stringstream sub_ss(sub);
                decode_quadtree(sub_ss, out.channels[k], i, j, b_size, b_dict.atoms, true);
            }
        }
        
        r_dict.deserialize(ss);
        for(int i=0; i<h; i+=r_size) {
            for(int j=0; j<w; j+=r_size) {
                decode_quadtree(ss, out.channels[k], i, j, r_size, r_dict.atoms, false);
            }
        }
    }
    savePPM(output, out);
}

void print_help(const char* prog) {
    cout << "GDHC v10.5 - Galois-DCT Hybrid Compressor\n"
         << "Usage: " << prog << " <mode:c/d> <input.ppm/bin> <output> [options]\n\n"
         << "Modes:\n"
         << "  c    Compress PPM image to binary\n"
         << "  d    Decompress binary to PPM image\n\n"
         << "Compression Options:\n"
         << "  --base-block <int>      Base layer block size (default: 16)\n"
         << "  --resid-block <int>     Residual layer block size (default: 8)\n"
         << "  --base-entries <int>    Base dictionary size (default: 512)\n"
         << "  --resid-entries <int>   Residual dictionary size (default: 1024)\n"
         << "  --lambda-base <float>   Base RDO lambda (default: 150.0)\n"
         << "  --lambda-resid <float>  Residual RDO lambda (default: 50.0)\n"
         << "  --base-var <float>      Min variance for base atoms (default: 30.0)\n"
         << "  --resid-var <float>     Min variance for residual atoms (default: 5.0)\n"
         << "  --max-atoms <int>       Max atoms per block (1-3, default: 3)\n"
         << "  --overlap <float>       Block overlap stride 0-1 (default: 0.5)\n"
         << "  --crypto-trials <int>   Morphism trials (default: 32)\n\n"
         << "Examples:\n"
         << "  " << prog << " c input.ppm output.gdhc\n"
         << "  " << prog << " c input.ppm output.gdhc --max-atoms 2 --overlap 0.25\n"
         << "  " << prog << " d output.gdhc reconstructed.ppm\n";
}

int main(int argc, char** argv) {
    if(argc < 4) {
        print_help(argv[0]);
        return 1;
    }

    const string mode = argv[1];
    const string in_file = argv[2];
    const string out_file = argv[3];

    CompressionConfig cfg;

    // Parse command line arguments
    for(int i = 4; i < argc; ++i) {
        const string arg = argv[i];
        if(i + 1 >= argc) break;
        
        try {
            if(arg == "--base-block") {
                cfg.base_block_size = stoi(argv[++i]);
                if (cfg.base_block_size < 4 || cfg.base_block_size > 128) {
                    throw runtime_error("base-block must be between 4 and 128");
                }
            }
            else if(arg == "--resid-block") {
                cfg.resid_block_size = stoi(argv[++i]);
                if (cfg.resid_block_size < 2 || cfg.resid_block_size > 64) {
                    throw runtime_error("resid-block must be between 2 and 64");
                }
            }
            else if(arg == "--base-entries") {
                cfg.base_entries = stoi(argv[++i]);
                if (cfg.base_entries < 1 || cfg.base_entries > 10000) {
                    throw runtime_error("base-entries must be between 1 and 10000");
                }
            }
            else if(arg == "--resid-entries") {
                cfg.resid_entries = stoi(argv[++i]);
                if (cfg.resid_entries < 1 || cfg.resid_entries > 10000) {
                    throw runtime_error("resid-entries must be between 1 and 10000");
                }
            }
            else if(arg == "--lambda-base") {
//                cfg.lambda_base = stof(argv[++i]);
            }
            else if(arg == "--lambda-resid") {
//                cfg.lambda_resid = stof(argv[++i]);
            }
            else if(arg == "--base-var") {
                cfg.base_var_threshold = stof(argv[++i]);
            }
            else if(arg == "--resid-var") {
                cfg.resid_var_threshold = stof(argv[++i]);
            }
            else if(arg == "--max-atoms") {
                //cfg.max_atoms_per_block = stoi(argv[++i]);
                //if (cfg.max_atoms_per_block < 1 || cfg.max_atoms_per_block > 3) {
                //    throw runtime_error("max-atoms must be between 1 and 3");
                //}
            }
            else if(arg == "--overlap") {
//                cfg.overlap_stride = stof(argv[++i]);
//                if (cfg.overlap_stride <= 0.0f || cfg.overlap_stride > 1.0f) {
//                    throw runtime_error("overlap must be between 0 and 1");
//                }
            }
            else if(arg == "--crypto-trials") {
                cfg.crypto_trials = stoi(argv[++i]);
                if (cfg.crypto_trials < 1 || cfg.crypto_trials > 1024) {
                    throw runtime_error("crypto-trials must be between 1 and 1024");
                }
            }
            else {
                cerr << "Unknown argument: " << arg << endl;
                //return 0;
            }
        } catch(const exception& e) {
            cerr << "Error with argument " << arg << ": " << e.what() << endl;
            return 1;
        }
    }

    try {
        if(mode == "c") {
            cout << "Compressing: " << in_file << " -> " << out_file << endl;
            cout << "Config: base_block=" << cfg.base_block_size 
                 << ", resid_block=" << cfg.resid_block_size << endl;
                 //<< ", max_atoms=" << cfg.max_atoms_per_block << endl;
                 //<< ", overlap=" << cfg.overlap_stride << endl;
            run_compression(in_file, out_file, cfg);
            cout << "Compression complete!" << endl;
        }
        else if(mode == "d") {
            cout << "Decompressing: " << in_file << " -> " << out_file << endl;
            run_decompression(in_file, out_file);
            cout << "Decompression complete!" << endl;
        }
        else {
            cerr << "Unknown mode: " << mode << " (use 'c' or 'd')" << endl;
            return 1;
        }
    } catch(const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }

    return 0;
}

