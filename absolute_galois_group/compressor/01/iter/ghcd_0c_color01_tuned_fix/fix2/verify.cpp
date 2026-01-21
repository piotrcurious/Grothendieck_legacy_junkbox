/**
 * Galois-DCT Hybrid Compressor (GDHC) - v10.5 "Residual-Verified + Associative"
 * IMPROVEMENTS:
 * - Dictionary training verifies actual residual reduction on training set
 * - Associative array: blocks can be placed anywhere with overlap
 * - Sparse coding: multiple atoms per region for better approximation
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
#include <set>
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
    int base_entries = 512;
    int resid_entries = 1024;
    float base_var_threshold = 30.0f;
    float resid_var_threshold = 5.0f;
    float lambda_base = 150.0f;
    float lambda_resid = 50.0f;
    int crypto_trials = 32;
    int max_atoms_per_block = 3;  // Sparse coding: multiple atoms
    float overlap_stride = 0.5f;  // 50% overlap for associative placement
};

enum QTNodeType : uint8_t { QT_LEAF = 0, QT_SPLIT = 1 };

#pragma pack(push,1)
struct FieldEntry {
    uint16_t id = 0;
    uint16_t morphism = 0;
    uint8_t offset = 0;
    uint8_t gain = 0;
};

struct AssociativeEntry {
    uint16_t row = 0;
    uint16_t col = 0;
    uint8_t num_atoms = 0;  // How many atoms used for this block
    FieldEntry atoms[3];     // Up to 3 atoms per block
};
#pragma pack(pop)

struct RDOStats {
    float ssd = FLT_MAX;
    float bits = 0;
    std::string stream;
};

// --- Math & Resizing ---

MatrixXf resize_matrix(const MatrixXf& src, int target_size) {
    if (src.rows() == target_size && src.cols() == target_size) return src;
    MatrixXf dst(target_size, target_size);
    float scale_r = (float)src.rows() / target_size;
    float scale_c = (float)src.cols() / target_size;

    for (int r = 0; r < target_size; ++r) {
        for (int c = 0; c < target_size; ++c) {
            float sr_f = r * scale_r;
            float sc_f = c * scale_c;
            int sr = std::min((int)sr_f, (int)src.rows() - 1);
            int sc = std::min((int)sc_f, (int)src.cols() - 1);
            
            if (scale_r >= 1.0f) {
                int r_limit = std::min((int)scale_r, (int)src.rows() - sr);
                int c_limit = std::min((int)scale_c, (int)src.cols() - sc);
                dst(r,c) = src.block(sr, sc, r_limit, c_limit).mean();
            } else {
                dst(r,c) = src(sr, sc);
            }
        }
    }
    return dst;
}

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

// --- Dictionary with Residual Verification ---

class ManifoldDictionary {
public:
    vector<MatrixXf> atoms;
    
    // Train with residual verification
    void train(const MatrixXf &data, int block_size, int max_entries, float min_var) {
        atoms.clear();
        if (data.rows() < block_size || data.cols() < block_size) return;

        // Extract all candidate blocks
        struct Candidate {
            MatrixXf normalized;
            MatrixXf original;
            float variance;
            int row, col;
        };
        vector<Candidate> candidates;
        
        for(int i=0; i <= data.rows()-block_size; i+=block_size/2) {
            for(int j=0; j <= data.cols()-block_size; j+=block_size/2) {
                MatrixXf b = data.block(i,j,block_size,block_size);
                float mu = b.mean();
                MatrixXf b_centered = b.array() - mu;
                float v = b_centered.squaredNorm();
                if (v < min_var) continue;
                
                MatrixXf canon = resize_matrix(b_centered, CANONICAL_SIZE);
                float n = canon.norm();
                if (n > 1e-5) {
                    candidates.push_back({canon/n, b_centered, v, i, j});
                }
            }
        }
        
        if (candidates.empty()) return;
        
        // Sort by variance (most structured patterns first)
        std::sort(candidates.begin(), candidates.end(), 
                  [](auto& a, auto& b){ return a.variance > b.variance; });
        
        // Greedy selection with residual verification
        MatrixXf residual_map = data;
        
        for(size_t i=0; i < candidates.size() && atoms.size() < (size_t)max_entries; ++i) {
            // Test if this atom actually reduces residuals
            float total_reduction = 0;
            int reduction_count = 0;
            
            for(int r=0; r <= residual_map.rows()-block_size; r+=block_size) {
                for(int c=0; c <= residual_map.cols()-block_size; c+=block_size) {
                    MatrixXf test_block = residual_map.block(r, c, block_size, block_size);
                    float mu = test_block.mean();
                    MatrixXf centered = test_block.array() - mu;
                    
                    // Try to approximate with this candidate
                    MatrixXf test_canon = resize_matrix(centered, CANONICAL_SIZE);
                    float tn = test_canon.norm();
                    if (tn < 1e-6f) continue;
                    test_canon /= tn;
                    
                    float best_corr = -1.0f;
                    for(uint16_t m=0; m < 16; ++m) {
                        MatrixXf transformed = CryptoMorphismEngine::apply(candidates[i].normalized, m);
                        float corr = (transformed.array() * test_canon.array()).sum();
                        best_corr = std::max(best_corr, std::abs(corr));
                    }
                    
                    if (best_corr > 0.3f) {  // Significant correlation
                        float before_err = centered.squaredNorm();
                        float after_err = before_err * (1 - best_corr * best_corr);
                        total_reduction += (before_err - after_err);
                        reduction_count++;
                    }
                }
            }
            
            // Add atom if it provides significant residual reduction
            float avg_reduction = reduction_count > 0 ? total_reduction / reduction_count : 0;
            if (avg_reduction > min_var * 0.1f || atoms.empty()) {
                // Check diversity from existing atoms
                bool is_diverse = true;
                for(const auto& existing : atoms) {
                    float similarity = (candidates[i].normalized.array() * existing.array()).sum();
                    if (std::abs(similarity) > 0.85f) {
                        is_diverse = false;
                        break;
                    }
                }
                
                if (is_diverse || atoms.empty()) {
                    atoms.push_back(candidates[i].normalized);
                    
                    // Update residual map by subtracting best approximations
                    for(int r=0; r <= residual_map.rows()-block_size; r+=block_size) {
                        for(int c=0; c <= residual_map.cols()-block_size; c+=block_size) {
                            MatrixXf test_block = residual_map.block(r, c, block_size, block_size);
                            float mu = test_block.mean();
                            MatrixXf centered = test_block.array() - mu;
                            
                            MatrixXf approx = best_approximation(centered, atoms.back(), block_size);
                            residual_map.block(r, c, block_size, block_size) -= approx;
                        }
                    }
                }
            }
        }
    }
    
    // Sparse coding: find multiple atoms to approximate target
    vector<FieldEntry> solve_sparse(const MatrixXf &target, vector<float> &scales, 
                                     int max_atoms, int max_trials) {
        vector<FieldEntry> entries;
        scales.clear();
        if(atoms.empty()) return entries;
        
        MatrixXf residual = target;
        
        for(int iter = 0; iter < max_atoms && !atoms.empty(); ++iter) {
            MatrixXf r_canon = resize_matrix(residual, CANONICAL_SIZE);
            float rnorm = r_canon.norm();
            if(rnorm < 1e-6f) break;
            r_canon /= rnorm;
            
            FieldEntry best;
            float best_corr = -1.0f;
            
            for(uint16_t i=0; i < (uint16_t)atoms.size(); ++i) {
                for(uint16_t m=0; m < 16; ++m) {
                    MatrixXf cand = CryptoMorphismEngine::apply(atoms[i], m);
                    float corr = (cand.array() * r_canon.array()).sum();
                    if(std::abs(corr) > best_corr) {
                        best_corr = std::abs(corr);
                        best.id = i;
                        best.morphism = m;
                    }
                }
            }
            
            if (best_corr < 0.15f) break;  // Not enough correlation
            
            float scale = best_corr * rnorm;
            if (std::abs(scale) < 1.0f) break;
            
            entries.push_back(best);
            scales.push_back(scale);
            
            // Subtract this approximation from residual
            MatrixXf atom_approx = resize_matrix(
                CryptoMorphismEngine::apply(atoms[best.id], best.morphism), 
                target.rows()
            );
            float n = atom_approx.norm();
            if (n > 1e-6f) {
                atom_approx *= (scale / n);
                residual -= atom_approx;
            }
        }
        
        return entries;
    }

private:
    MatrixXf best_approximation(const MatrixXf& target, const MatrixXf& atom, int size) {
        MatrixXf t_canon = resize_matrix(target, CANONICAL_SIZE);
        float tn = t_canon.norm();
        if (tn < 1e-6f) return MatrixXf::Zero(size, size);
        t_canon /= tn;
        
        float best_corr = -1.0f;
        uint16_t best_m = 0;
        for(uint16_t m=0; m < 16; ++m) {
            MatrixXf transformed = CryptoMorphismEngine::apply(atom, m);
            float corr = (transformed.array() * t_canon.array()).sum();
            if (std::abs(corr) > best_corr) {
                best_corr = std::abs(corr);
                best_m = m;
            }
        }
        
        MatrixXf result = resize_matrix(CryptoMorphismEngine::apply(atom, best_m), size);
        float scale = best_corr * tn;
        float n = result.norm();
        if (n > 1e-6f) result *= (scale / n);
        return result;
    }
};

// --- Image I/O (unchanged) ---

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

// --- Associative Array Compression ---

void compress_associative(const MatrixXf &src, ManifoldDictionary &dict, 
                          MatrixXf &recon, stringstream &stream,
                          int block_size, const CompressionConfig &cfg) {
    recon = MatrixXf::Zero(src.rows(), src.cols());
    MatrixXf residual = src;
    
    int stride = std::max(1, (int)(block_size * cfg.overlap_stride));
    vector<AssociativeEntry> entries;
    
    // Greedy block placement with overlap
    for(int i=0; i <= src.rows()-block_size; i+=stride) {
        for(int j=0; j <= src.cols()-block_size; j+=stride) {
            MatrixXf block = residual.block(i, j, block_size, block_size);
            float mu = block.mean();
            MatrixXf centered = block.array() - mu;
            
            if (centered.squaredNorm() < 10.0f) continue;  // Skip near-flat blocks
            
            // Sparse coding: find multiple atoms
            vector<float> scales;
            auto atoms = dict.solve_sparse(centered, scales, cfg.max_atoms_per_block, cfg.crypto_trials);
            
            if (atoms.empty()) continue;
            
            AssociativeEntry entry;
            entry.row = i;
            entry.col = j;
            entry.num_atoms = std::min((int)atoms.size(), 3);
            
            MatrixXf reconstruction = MatrixXf::Zero(block_size, block_size);
            float q_step = 1.0f;
            
            for(int a=0; a < entry.num_atoms; ++a) {
                entry.atoms[a] = atoms[a];
                entry.atoms[a].offset = (uint8_t)std::clamp(mu + 128.0f, 0.0f, 255.0f);
                entry.atoms[a].gain = (uint8_t)std::clamp(std::abs(scales[a]) / q_step, 0.0f, 255.0f);
                
                // Reconstruct
                if (entry.atoms[a].id < dict.atoms.size() && entry.atoms[a].gain > 0) {
                    MatrixXf atom = resize_matrix(
                        CryptoMorphismEngine::apply(dict.atoms[entry.atoms[a].id], entry.atoms[a].morphism),
                        block_size
                    );
                    float n = atom.norm();
                    if (n > 1e-6f) {
                        float actual_scale = scales[a] < 0 ? -entry.atoms[a].gain * q_step : entry.atoms[a].gain * q_step;
                        reconstruction += atom * (actual_scale / n);
                    }
                }
            }
            
            reconstruction.array() += (entry.atoms[0].offset - 128);
            
            // Update residual and reconstruction
            residual.block(i, j, block_size, block_size) -= reconstruction;
            recon.block(i, j, block_size, block_size) += reconstruction;
            
            entries.push_back(entry);
        }
    }
    
    // Write entries
    uint32_t num_entries = entries.size();
    stream.write((char*)&num_entries, 4);
    for(const auto& e : entries) {
        stream.write((char*)&e, sizeof(AssociativeEntry));
    }
}

void decompress_associative(stringstream &ss, MatrixXf &target, int block_size,
                            const vector<MatrixXf> &atoms) {
    target = MatrixXf::Zero(target.rows(), target.cols());
    
    uint32_t num_entries;
    ss.read((char*)&num_entries, 4);
    
    for(uint32_t e=0; e < num_entries; ++e) {
        AssociativeEntry entry;
        ss.read((char*)&entry, sizeof(AssociativeEntry));
        
        if (entry.row >= target.rows() || entry.col >= target.cols()) continue;
        
        int h = std::min(block_size, (int)target.rows() - entry.row);
        int w = std::min(block_size, (int)target.cols() - entry.col);
        
        MatrixXf reconstruction = MatrixXf::Zero(h, w);
        float q_step = 1.0f;
        
        for(int a=0; a < entry.num_atoms; ++a) {
            if (entry.atoms[a].id < atoms.size() && entry.atoms[a].gain > 0) {
                MatrixXf atom = resize_matrix(
                    CryptoMorphismEngine::apply(atoms[entry.atoms[a].id], entry.atoms[a].morphism),
                    block_size
                ).block(0, 0, h, w);
                
                float n = atom.norm();
                if (n > 1e-6f) {
                    reconstruction += atom * (entry.atoms[a].gain * q_step / n);
                }
            }
        }
        
        reconstruction.array() += (entry.atoms[0].offset - 128);
        target.block(entry.row, entry.col, h, w) += reconstruction;
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
        
        // Train base dictionary with residual verification
        b_dict.train(img.channels[k], cfg.base_block_size, cfg.base_entries, cfg.base_var_threshold);
        uint32_t n_b = b_dict.atoms.size();
        bitstream.write((char*)&n_b, 4);
        for(auto &a : b_dict.atoms) bitstream.write((char*)a.data(), a.size()*sizeof(float));

        // Compress with associative placement
        compress_associative(img.channels[k], b_dict, recon_full.channels[k], bitstream,
                           cfg.base_block_size, cfg);

        // Residual layer
        MatrixXf residual = img.channels[k] - recon_full.channels[k];
        r_dict.train(residual, cfg.resid_block_size, cfg.resid_entries, cfg.resid_var_threshold);
        uint32_t n_r = r_dict.atoms.size();
        bitstream.write((char*)&n_r, 4);
        for(auto &a : r_dict.atoms) bitstream.write((char*)a.data(), a.size()*sizeof(float));

        MatrixXf resid_recon = MatrixXf::Zero(img.height, img.width);
        compress_associative(residual, r_dict, resid_recon, bitstream,
                           cfg.resid_block_size, cfg);
        
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
        
        uint32_t n_b; ss.read((char*)&n_b, 4);
        vector<MatrixXf> b_atoms(n_b, MatrixXf::Zero(CANONICAL_SIZE, CANONICAL_SIZE));
        for(auto &a : b_atoms) ss.read((char*)a.data(), a.size()*sizeof(float));
        
        decompress_associative(ss, out.channels[k], b_size, b_atoms);
        
        uint32_t n_r; ss.read((char*)&n_r, 4);
        vector<MatrixXf> r_atoms(n_r, MatrixXf::Zero(CANONICAL_SIZE, CANONICAL_SIZE));
        for(auto &a : r_atoms) ss.read((char*)a.data(), a.size()*sizeof(float));
        
        MatrixXf resid_recon(h, w);
        decompress_associative(ss, resid_recon, r_size, r_atoms);
        out.channels[k] += resid_recon;
    }
    savePPM(output, out);
}

void print_help(const char* prog) {
    cout << "Usage: " << prog << " <mode:c/d> <input.ppm/bin> <output> [options]\n"
         << "Options:\n"
         << "  --base-block <int>      Size of base layer blocks (default: 16)\n"
         << "  --resid-block <int>     Size of residual blocks (default: 8)\n"
         << "  --base-entries <int>    Dictionary size for base layer (default: 512)\n"
         << "  --resid-entries <int>   Dictionary size for residual layer (default: 1024)\n"
         << "  --lambda-base <float>   RDO Lambda for base (default: 150.0)\n"
         << "  --lambda-resid <float>  RDO Lambda for residual (default: 50.0)\n"
         << "  --max-atoms <int>       Max atoms per block (default: 3)\n"
         << "  --overlap <float>       Overlap stride factor 0-1 (default: 0.5)\n";
}

int main(int argc, char** argv) {
    if(argc < 4) { print_help(argv[0]); return 1; }

    string mode = argv[1], in_file = argv[2], out_file = argv[3];
    CompressionConfig cfg;

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
            else if(arg == "--max-atoms") cfg.max_atoms_per_block = stoi(argv[++i]);
            else if(arg == "--overlap") cfg.overlap_stride = stof(argv[++i]);
        } catch(...) {
            cerr << "Invalid value for argument: " << arg << endl;
            return 1;
        }
    }

    if(mode == "c") run_compression(in_file, out_file, cfg);
    else if(mode == "d") run_decompression(in_file, out_file);
    else { cerr << "Unknown mode: " << mode << endl; return 1; }

    return 0;
}
