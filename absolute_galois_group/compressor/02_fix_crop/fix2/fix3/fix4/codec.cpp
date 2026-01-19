#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <string>
#include <stdexcept>
#include <Eigen/Dense>
#include <memory>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/zlib.hpp>

using namespace Eigen;
using namespace std;

// --- Enhanced Architecture Configuration ---
const int POLY_ORDER = 4;
const int DICT_SIZE = 128;
const int MIN_BLOCK = 8;      // Increased from 4
const int MAX_BLOCK = 64;
const float VAR_THRESHOLD = 800.0f;  // Adjusted for 8x8 minimum
const float HIGH_FREQ_THRESHOLD = 2000.0f;  // For Galois block activation
const int MAX_RESIDUAL_LEVELS = 3;  // Hierarchical refinement depth

// Quality/Compression tradeoff (0-10)
const int QUALITY_LEVEL = 7;

// --- Mathematical Abstractions ---

class OrthogonalKernel {
public:
    virtual float eval(float x, float y, int basis_idx) const = 0;
    virtual int basis_size() const = 0;
    virtual ~OrthogonalKernel() = default;
};

// Cached Legendre Kernel with precomputed basis matrices
class Legendre2DKernel : public OrthogonalKernel {
    int order;
    int dim;
    
    // Cache for common block sizes (powers of 2: 8, 16, 32, 64)
    mutable map<int, MatrixXf> basis_cache;
    
public:
    Legendre2DKernel(int n) : order(n), dim((n+1)*(n+1)) {
        precompute_common_sizes();
    }
    
    void precompute_common_sizes() {
        cout << "[*] Precomputing basis matrices..." << endl;
        for (int size : {8, 16, 32, 64}) {
            get_basis_matrix(size, size);
        }
    }

    float legendre(int n, float x) const {
        if (n == 0) return 1.0f;
        if (n == 1) return x;
        float p0 = 1.0f, p1 = x, p2;
        for (int i = 2; i <= n; ++i) {
            p2 = ((2 * i - 1) * x * p1 - (i - 1) * p0) / i;
            p0 = p1; p1 = p2;
        }
        return p1;
    }

    float eval(float x, float y, int k) const override {
        int side = order + 1;
        int i = k / side;
        int j = k % side;
        float norm_x = sqrt((2.0f * i + 1.0f) / 2.0f);
        float norm_y = sqrt((2.0f * j + 1.0f) / 2.0f);
        return (legendre(i, x) * norm_x) * (legendre(j, y) * norm_y);
    }
    
    // Get or create cached basis matrix for reconstruction
    const MatrixXf& get_basis_matrix(int h, int w) const {
        int key = (h << 16) | w;
        auto it = basis_cache.find(key);
        if (it != basis_cache.end()) {
            return it->second;
        }
        
        // Compute basis matrix: [h*w x basis_size]
        MatrixXf basis(h * w, dim);
        for (int r = 0; r < h; ++r) {
            float y = -1.0f + 2.0f * (r + 0.5f) / h;
            for (int c = 0; c < w; ++c) {
                float x = -1.0f + 2.0f * (c + 0.5f) / w;
                int idx = r * w + c;
                for (int k = 0; k < dim; ++k) {
                    basis(idx, k) = eval(x, y, k);
                }
            }
        }
        basis_cache[key] = basis;
        return basis_cache[key];
    }

    int basis_size() const override { return dim; }
};

unique_ptr<Legendre2DKernel> kernel = make_unique<Legendre2DKernel>(POLY_ORDER);

// --- Galois-Inspired High-Frequency Block ---
// Uses integer lattice structure for sharp edges/textures
struct GaloisBlock {
    vector<int8_t> deltas;  // First-order differences
    uint8_t predictor_mode; // 0=horizontal, 1=vertical, 2=diagonal
    
    static GaloisBlock encode(const MatrixXf& block) {
        int h = block.rows(), w = block.cols();
        GaloisBlock gb;
        
        // Choose best predictor direction
        float h_energy = 0, v_energy = 0, d_energy = 0;
        for (int r = 0; r < h; ++r) {
            for (int c = 1; c < w; ++c) {
                h_energy += abs(block(r, c) - block(r, c-1));
            }
        }
        for (int r = 1; r < h; ++r) {
            for (int c = 0; c < w; ++c) {
                v_energy += abs(block(r, c) - block(r-1, c));
            }
        }
        for (int r = 1; r < h && r < w; ++r) {
            for (int c = 1; c < w && c < h; ++c) {
                d_energy += abs(block(r, c) - block(r-1, c-1));
            }
        }
        
        gb.predictor_mode = (h_energy < v_energy) ? 
                           ((h_energy < d_energy) ? 0 : 2) : 
                           ((v_energy < d_energy) ? 1 : 2);
        
        // Encode deltas
        gb.deltas.reserve(h * w);
        for (int r = 0; r < h; ++r) {
            for (int c = 0; c < w; ++c) {
                float pred = 0;
                if (gb.predictor_mode == 0 && c > 0) pred = block(r, c-1);
                else if (gb.predictor_mode == 1 && r > 0) pred = block(r-1, c);
                else if (gb.predictor_mode == 2 && r > 0 && c > 0) pred = block(r-1, c-1);
                else pred = 128.0f; // Default predictor
                
                int delta = clamp((int)(block(r, c) - pred), -127, 127);
                gb.deltas.push_back((int8_t)delta);
            }
        }
        return gb;
    }
    
    static MatrixXf decode(const GaloisBlock& gb, int h, int w) {
        MatrixXf block(h, w);
        int idx = 0;
        for (int r = 0; r < h; ++r) {
            for (int c = 0; c < w; ++c) {
                float pred = 0;
                if (gb.predictor_mode == 0 && c > 0) pred = block(r, c-1);
                else if (gb.predictor_mode == 1 && r > 0) pred = block(r-1, c);
                else if (gb.predictor_mode == 2 && r > 0 && c > 0) pred = block(r-1, c-1);
                else pred = 128.0f;
                
                block(r, c) = clamp(pred + gb.deltas[idx++], 0.0f, 255.0f);
            }
        }
        return block;
    }
};

// --- Data Structures ---

struct ResidualLayer {
    vector<int8_t> residuals;
    float quantization_step;
};

struct CompressedBlock {
    uint8_t atom_index;
    int16_t scale;        // Signed 16-bit for better dynamic range
    uint8_t offset;
    uint16_t size_info;
    bool use_galois;      // Flag for high-frequency blocks
    
    // Optional Galois data
    unique_ptr<GaloisBlock> galois_data;
    
    // Hierarchical residuals
    vector<ResidualLayer> residuals;
};

struct CompressedData {
    int orig_width, orig_height;
    int padded_width, padded_height;
    MatrixXf poly_dictionary;
    vector<bool> split_tree;
    vector<CompressedBlock> blocks;
};

// --- Enhanced Functional Analysis ---

VectorXf project_onto_basis(const MatrixXf& block) {
    int h = block.rows(), w = block.cols();
    int b_size = kernel->basis_size();
    VectorXf coeffs = VectorXf::Zero(b_size);

    for (int r = 0; r < h; ++r) {
        float y = -1.0f + 2.0f * (r + 0.5f) / h;
        for (int c = 0; c < w; ++c) {
            float x = -1.0f + 2.0f * (c + 0.5f) / w;
            float val = block(r, c);
            
            for (int k = 0; k < b_size; ++k) {
                coeffs(k) += val * kernel->eval(x, y, k);
            }
        }
    }
    coeffs *= (4.0f / (w * h));
    return coeffs;
}

MatrixXf reconstruction_map(const VectorXf& coeffs, int h, int w) {
    const MatrixXf& basis = kernel->get_basis_matrix(h, w);
    VectorXf flat = basis * coeffs;
    return Map<MatrixXf>(flat.data(), h, w);
}

struct Image { 
    int width, height; 
    MatrixXf data; 
};

Image loadPGM(const string& filename) {
    ifstream file(filename, ios::binary);
    if (!file) throw runtime_error("Cannot open file: " + filename);
    
    string magic; 
    file >> magic;
    if (magic != "P5") throw runtime_error("Invalid PGM format");
    
    int w, h, maxval;
    file >> w >> h >> maxval;
    if (w <= 0 || h <= 0 || w > 16384 || h > 16384) {
        throw runtime_error("Invalid image dimensions");
    }
    if (maxval != 255) throw runtime_error("Only 8-bit PGM supported");
    
    file.ignore(1);
    
    Image img{w, h, MatrixXf(h, w)};
    vector<uint8_t> buf(w * h);
    file.read((char*)buf.data(), buf.size());
    
    if (!file) throw runtime_error("Failed to read image data");
    
    for (int i = 0; i < h * w; ++i) {
        img.data(i / w, i % w) = buf[i];
    }
    return img;
}

void savePGM(const string& filename, const Image& img) {
    ofstream file(filename, ios::binary);
    if (!file) throw runtime_error("Cannot write file: " + filename);
    
    file << "P5\n" << img.width << " " << img.height << "\n255\n";
    vector<uint8_t> buf(img.width * img.height);
    
    for (int i = 0; i < img.height; ++i) {
        for (int j = 0; j < img.width; ++j) {
            buf[i * img.width + j] = (uint8_t)clamp(img.data(i, j), 0.f, 255.f);
        }
    }
    
    file.write((char*)buf.data(), buf.size());
    if (!file) throw runtime_error("Failed to write image data");
}

// --- Enhanced Dictionary Learning ---

MatrixXf learn_polynomial_dictionary(const Image& img) {
    cout << "[*] Learning Polynomial Manifold..." << endl;
    vector<VectorXf> samples;
    
    mt19937 rng(12345);
    int steps[] = {8, 16, 32, 64};
    
    for (int s : steps) {
        for (int i = 0; i + s <= img.height; i += s/2) {
            for (int j = 0; j + s <= img.width; j += s/2) {
                if (rng() % 3 != 0) continue;
                
                MatrixXf blk = img.data.block(i, j, s, s);
                float mean = blk.mean();
                blk.array() -= mean;
                
                VectorXf coeffs = project_onto_basis(blk);
                float n = coeffs.norm();
                if (n > 1.0f) {
                    coeffs /= n;
                    samples.push_back(coeffs);
                }
            }
        }
    }
    
    if (samples.empty()) {
        throw runtime_error("No samples collected for dictionary learning");
    }
    
    int dim = kernel->basis_size();
    MatrixXf dict = MatrixXf::Random(dim, DICT_SIZE);
    
    for (int k = 0; k < DICT_SIZE; ++k) dict.col(k).normalize();
    
    cout << "    Clustering " << samples.size() << " polynomial vectors..." << endl;

    float prev_shift = 1e10;
    for (int iter = 0; iter < 30; ++iter) {
        MatrixXf centroids = MatrixXf::Zero(dim, DICT_SIZE);
        vector<int> counts(DICT_SIZE, 0);
        
        for (size_t i = 0; i < samples.size(); ++i) {
            VectorXf::Index best;
            (dict.transpose() * samples[i]).maxCoeff(&best);
            centroids.col(best) += samples[i];
            counts[best]++;
        }
        
        float shift = 0;
        for (int k = 0; k < DICT_SIZE; ++k) {
            if (counts[k] > 0) {
                centroids.col(k) /= counts[k];
                centroids.col(k).normalize();
                shift += (centroids.col(k) - dict.col(k)).norm();
                dict.col(k) = centroids.col(k);
            } else {
                // Reinitialize with furthest sample
                float max_dist = -1;
                int furthest = 0;
                for (size_t i = 0; i < samples.size(); ++i) {
                    float min_sim = 1.0f;
                    for (int j = 0; j < DICT_SIZE; ++j) {
                        min_sim = min(min_sim, abs(dict.col(j).dot(samples[i])));
                    }
                    if (min_sim > max_dist) {
                        max_dist = min_sim;
                        furthest = i;
                    }
                }
                dict.col(k) = samples[furthest];
            }
        }
        
        if (iter > 5 && shift > prev_shift * 0.99f) {
            cout << "    Converged after " << iter << " iterations" << endl;
            break;
        }
        prev_shift = shift;
    }
    
    return dict;
}

// --- Hierarchical Residual Encoding ---

vector<ResidualLayer> encode_residuals(const MatrixXf& original, 
                                       const MatrixXf& approx,
                                       int max_levels) {
    vector<ResidualLayer> layers;
    MatrixXf current_residual = original - approx;
    
    for (int level = 0; level < max_levels; ++level) {
        float quant_step = 16.0f / pow(2, level);  // Exponential refinement
        ResidualLayer layer;
        layer.quantization_step = quant_step;
        
        float max_improvement = 0;
        for (int i = 0; i < current_residual.size(); ++i) {
            int quantized = (int)round(current_residual(i) / quant_step);
            quantized = clamp(quantized, -127, 127);
            layer.residuals.push_back((int8_t)quantized);
            max_improvement = max(max_improvement, abs(current_residual(i)));
        }
        
        // Early stop if residuals are negligible
        if (max_improvement < quant_step * 1.5f) break;
        
        // Update residual for next level
        for (int i = 0; i < current_residual.size(); ++i) {
            current_residual(i) -= layer.residuals[i] * quant_step;
        }
        
        layers.push_back(layer);
    }
    
    return layers;
}

MatrixXf decode_residuals(const MatrixXf& base, 
                          const vector<ResidualLayer>& layers,
                          int h, int w) {
    MatrixXf result = base;
    for (const auto& layer : layers) {
        int idx = 0;
        for (int i = 0; i < h * w; ++i) {
            result(i) += layer.residuals[idx++] * layer.quantization_step;
        }
    }
    return result;
}

// --- Enhanced Quadtree Processing ---

void process_quadrant(const Ref<const MatrixXf>& img_block, 
                     const MatrixXf& dict, 
                     CompressedData& storage) {
    int h = img_block.rows(), w = img_block.cols();
    float mean = img_block.mean();
    float var = (img_block.array() - mean).square().mean();
    
    // Measure high-frequency content
    float edge_energy = 0;
    for (int r = 1; r < h; ++r) {
        for (int c = 1; c < w; ++c) {
            edge_energy += abs(img_block(r,c) - img_block(r-1,c)) + 
                          abs(img_block(r,c) - img_block(r,c-1));
        }
    }
    edge_energy /= (h * w);
    
    bool can_split = (h >= 2*MIN_BLOCK && w >= 2*MIN_BLOCK);
    bool must_split = (h > MAX_BLOCK || w > MAX_BLOCK);
    
    if (must_split || (can_split && var > VAR_THRESHOLD)) {
        storage.split_tree.push_back(true);
        int hh = h/2, hw = w/2;
        process_quadrant(img_block.block(0, 0, hh, hw), dict, storage);
        process_quadrant(img_block.block(0, hw, hh, w-hw), dict, storage);
        process_quadrant(img_block.block(hh, 0, h-hh, hw), dict, storage);
        process_quadrant(img_block.block(hh, hw, h-hh, w-hw), dict, storage);
    } else {
        storage.split_tree.push_back(false);
        
        CompressedBlock cb;
        cb.size_info = (uint16_t)((h << 8) | w);
        cb.offset = (uint8_t)clamp(mean, 0.f, 255.f);
        
        // Decide: Polynomial or Galois encoding?
        bool use_galois = (edge_energy > HIGH_FREQ_THRESHOLD / (h * w));
        cb.use_galois = use_galois;
        
        if (use_galois) {
            // High-frequency: Use Galois integer prediction
            cb.galois_data = make_unique<GaloisBlock>(GaloisBlock::encode(img_block));
            cb.atom_index = 255;  // Sentinel value
            cb.scale = 0;
        } else {
            // Smooth region: Use polynomial approximation
            MatrixXf ac_block = img_block.array() - mean;
            VectorXf coeffs = project_onto_basis(ac_block);
            
            float norm = coeffs.norm();
            if (norm < 1e-4f) norm = 1.0f;
            VectorXf normalized_coeffs = coeffs / norm;
            
            VectorXf::Index best_k;
            (dict.transpose() * normalized_coeffs).maxCoeff(&best_k);
            
            float final_scale = dict.col(best_k).dot(coeffs);
            
            // Enhanced scale encoding: signed 16-bit with log compression
            bool is_negative = final_scale < 0;
            float abs_scale = abs(final_scale);
            
            int16_t encoded_scale = 0;
            if (abs_scale > 1e-3f) {
                float log_scale = log2(abs_scale);
                encoded_scale = (int16_t)clamp(
                    4096.0f + log_scale * 512.0f,
                    0.0f, 32767.0f
                );
                if (is_negative) encoded_scale = -encoded_scale;
            }
            
            cb.atom_index = (uint8_t)best_k;
            cb.scale = encoded_scale;
            
            // Hierarchical residual encoding for quality
            if (QUALITY_LEVEL >= 5) {
                MatrixXf approx = reconstruction_map(dict.col(best_k) * final_scale, h, w);
                approx.array() += mean;
                
                int residual_levels = (QUALITY_LEVEL >= 8) ? 3 : 
                                     (QUALITY_LEVEL >= 6) ? 2 : 1;
                cb.residuals = encode_residuals(img_block, approx, residual_levels);
            }
        }
        
        storage.blocks.push_back(move(cb));
    }
}

// --- Compression Pipeline ---

void compress(const string& fin, const string& fout) {
    cout << "[*] Loading image: " << fin << endl;
    Image img = loadPGM(fin);
    
    int oh = img.height, ow = img.width;
    int ph = 1, pw = 1;
    while (ph < oh) ph *= 2;
    while (pw < ow) pw *= 2;
    
    MatrixXf data = MatrixXf::Zero(ph, pw);
    data.block(0, 0, oh, ow) = img.data;
    
    MatrixXf dict = learn_polynomial_dictionary(img);
    
    CompressedData cd;
    cd.orig_height = oh; 
    cd.orig_width = ow;
    cd.padded_height = ph; 
    cd.padded_width = pw;
    cd.poly_dictionary = dict;
    
    cout << "[*] Encoding image blocks..." << endl;
    process_quadrant(data, dict, cd);
    
    cout << "[*] Writing compressed file: " << fout << endl;
    ofstream file(fout, ios::binary);
    if (!file) throw runtime_error("Cannot write output file");
    
    boost::iostreams::filtering_streambuf<boost::iostreams::output> out;
    out.push(boost::iostreams::zlib_compressor());
    out.push(file);
    ostream os(&out);
    
    // Write header
    uint32_t magic = 0x504F4C59;  // 'POLY'
    os.write((char*)&magic, 4);
    os.write((char*)&oh, 4); 
    os.write((char*)&ow, 4);
    os.write((char*)&ph, 4); 
    os.write((char*)&pw, 4);
    
    // Write dictionary
    os.write((char*)dict.data(), dict.size() * sizeof(float));
    
    // Write tree
    int ts = cd.split_tree.size();
    os.write((char*)&ts, 4);
    vector<uint8_t> pt((ts + 7) / 8, 0);
    for (int i = 0; i < ts; ++i) {
        if (cd.split_tree[i]) pt[i/8] |= (1 << (i%8));
    }
    os.write((char*)pt.data(), pt.size());
    
    // Write blocks
    int bc = cd.blocks.size();
    os.write((char*)&bc, 4);
    
    for (const auto& block : cd.blocks) {
        os.write((char*)&block.atom_index, 1);
        os.write((char*)&block.scale, 2);
        os.write((char*)&block.offset, 1);
        os.write((char*)&block.size_info, 2);
        os.write((char*)&block.use_galois, 1);
        
        if (block.use_galois) {
            os.write((char*)&block.galois_data->predictor_mode, 1);
            int delta_count = block.galois_data->deltas.size();
            os.write((char*)&delta_count, 4);
            os.write((char*)block.galois_data->deltas.data(), delta_count);
        } else {
            uint8_t num_residuals = block.residuals.size();
            os.write((char*)&num_residuals, 1);
            for (const auto& layer : block.residuals) {
                os.write((char*)&layer.quantization_step, sizeof(float));
                int res_count = layer.residuals.size();
                os.write((char*)&res_count, 4);
                os.write((char*)layer.residuals.data(), res_count);
            }
        }
    }
    
    cout << "[*] Compression complete!" << endl;
    cout << "    Blocks: " << bc << " (Galois: " 
         << count_if(cd.blocks.begin(), cd.blocks.end(), 
                    [](const CompressedBlock& b) { return b.use_galois; })
         << ")" << endl;
}

// --- Decompression Pipeline ---

struct Reader {
    vector<bool> tree;
    vector<CompressedBlock> blocks;
    int ti = 0, bi = 0;
};

void reconstruct(Ref<MatrixXf> blk, const MatrixXf& dict, Reader& r) {
    if (r.ti >= (int)r.tree.size()) {
        throw runtime_error("Tree index out of bounds during reconstruction");
    }
    
    if (r.tree[r.ti++]) {
        int h = blk.rows(), w = blk.cols();
        reconstruct(blk.block(0, 0, h/2, w/2), dict, r);
        reconstruct(blk.block(0, w/2, h/2, w-w/2), dict, r);
        reconstruct(blk.block(h/2, 0, h-h/2, w/2), dict, r);
        reconstruct(blk.block(h/2, w/2, h-h/2, w-w/2), dict, r);
    } else {
        if (r.bi >= (int)r.blocks.size()) {
            throw runtime_error("Block index out of bounds during reconstruction");
        }
        
        const CompressedBlock& cb = r.blocks[r.bi++];
        int bh = (cb.size_info >> 8) & 0xFF;
        int bw = cb.size_info & 0xFF;
        
        if (cb.use_galois) {
            // Decode Galois block
            blk = GaloisBlock::decode(*cb.galois_data, bh, bw);
        } else {
            // Decode polynomial block
            VectorXf coeffs = dict.col(cb.atom_index);
            
            // Decode scale
            float scale = 0;
            if (cb.scale != 0) {
                bool is_negative = cb.scale < 0;
                int16_t abs_encoded = is_negative ? -cb.scale : cb.scale;
                float log_scale = (abs_encoded - 4096.0f) / 512.0f;
                scale = pow(2.0f, log_scale);
                if (is_negative) scale = -scale;
            }
            
            coeffs *= scale;
            MatrixXf res = reconstruction_map(coeffs, bh, bw);
            res.array() += cb.offset;
            
            // Apply residuals
            if (!cb.residuals.empty()) {
                res = decode_residuals(res, cb.residuals, bh, bw);
            }
            
            blk = res;
        }
    }
}

void decompress(const string& fin, const string& fout) {
    cout << "[*] Reading compressed file: " << fin << endl;
    ifstream file(fin, ios::binary);
    if (!file) throw runtime_error("Cannot open compressed file");
    
    boost::iostreams::filtering_streambuf<boost::iostreams::input> in;
    in.push(boost::iostreams::zlib_decompressor());
    in.push(file);
    istream is(&in);
    
    uint32_t magic;
    is.read((char*)&magic, 4);
    if (magic != 0x504F4C59) throw runtime_error("Invalid file format");
    
    int oh, ow, ph, pw;
    is.read((char*)&oh, 4); 
    is.read((char*)&ow, 4);
    is.read((char*)&ph, 4); 
    is.read((char*)&pw, 4);
    
    MatrixXf dict(kernel->basis_size(), DICT_SIZE);
    is.read((char*)dict.data(), dict.size() * 4);
    if (!is) throw runtime_error("Failed to read dictionary");
    
    int ts; 
    is.read((char*)&ts, 4);
    vector<uint8_t> pt((ts + 7) / 8);
    is.read((char*)pt.data(), pt.size());
    
    Reader r;
    r.tree.resize(ts);
    for (int i = 0; i < ts; ++i) {
        r.tree[i] = (pt[i/8] >> (i%8)) & 1;
    }
    
    int bc;
