#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <string>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/zlib.hpp>

/**
 * Topological Algebraic Compressor (TAC) - Hierarchical Residual Edition
 * Features:
 * 1. Adaptive Quadtree with algebraic summation of parent+child atoms
 * 2. Children encode RESIDUALS from parent reconstruction
 * 3. Dictionary atoms form an additive group - shapes can be reused and summed
 * 
 * Mathematical Foundation:
 *   Block(x,y) = Parent_Atom + α₁·Child1_Atom + α₂·Child2_Atom + ...
 * 
 * Compile:
 * g++ -O3 -I /usr/include/eigen3 tac_compressor.cpp -o tac -lboost_iostreams -lz -fopenmp
 */

using namespace Eigen;
using namespace std;

// --- Tuning Parameters ---
const int CANONICAL_SIZE = 8;
const int CANONICAL_DIM = CANONICAL_SIZE * CANONICAL_SIZE;
const int DICT_SIZE = 256; 
const int MIN_BLOCK = 4;
const int MAX_BLOCK = 64;
const float VAR_THRESHOLD = 150.0f;
const float RESIDUAL_THRESHOLD = 50.0f;  // Stop splitting if residual is small

struct Image {
    int width, height;
    MatrixXf data;
};

struct CompressedData {
    int width, height;
    MatrixXf dictionary;
    std::vector<bool> split_tree;
    std::vector<uint8_t> indices;
    std::vector<int8_t> scales;      // Signed for residuals
    std::vector<int8_t> offsets;     // Signed for residuals
};

// --- Helper: Resampling ---
VectorXf canonicalize(const MatrixXf& block) {
    if (block.rows() == CANONICAL_SIZE && block.cols() == CANONICAL_SIZE) {
        return Map<const VectorXf>(block.data(), CANONICAL_DIM);
    }
    
    MatrixXf resized(CANONICAL_SIZE, CANONICAL_SIZE);
    float r_step = (float)block.rows() / CANONICAL_SIZE;
    float c_step = (float)block.cols() / CANONICAL_SIZE;

    for (int i = 0; i < CANONICAL_SIZE; ++i) {
        for (int j = 0; j < CANONICAL_SIZE; ++j) {
            float src_r = i * r_step;
            float src_c = j * c_step;
            
            int r0 = (int)src_r;
            int c0 = (int)src_c;
            int r1 = std::min(r0 + 1, (int)block.rows() - 1);
            int c1 = std::min(c0 + 1, (int)block.cols() - 1);
            
            float dr = src_r - r0;
            float dc = src_c - c0;
            
            resized(i, j) = (1-dr)*(1-dc)*block(r0,c0) + 
                           (1-dr)*dc*block(r0,c1) +
                           dr*(1-dc)*block(r1,c0) +
                           dr*dc*block(r1,c1);
        }
    }
    return Map<VectorXf>(resized.data(), CANONICAL_DIM);
}

MatrixXf reconstruction_map(const VectorXf& vec, int target_rows, int target_cols) {
    MatrixXf canonical = Map<const MatrixXf>(vec.data(), CANONICAL_SIZE, CANONICAL_SIZE);
    
    if (target_rows == CANONICAL_SIZE && target_cols == CANONICAL_SIZE) {
        return canonical;
    }

    MatrixXf out(target_rows, target_cols);
    float r_ratio = (float)(CANONICAL_SIZE - 1) / std::max(1, target_rows - 1);
    float c_ratio = (float)(CANONICAL_SIZE - 1) / std::max(1, target_cols - 1);

    for (int i = 0; i < target_rows; ++i) {
        for (int j = 0; j < target_cols; ++j) {
            float src_r = i * r_ratio;
            float src_c = j * c_ratio;
            
            int r0 = (int)src_r;
            int c0 = (int)src_c;
            int r1 = std::min(r0 + 1, CANONICAL_SIZE - 1);
            int c1 = std::min(c0 + 1, CANONICAL_SIZE - 1);
            
            float dr = src_r - r0;
            float dc = src_c - c0;
            
            out(i, j) = (1-dr)*(1-dc)*canonical(r0,c0) + 
                       (1-dr)*dc*canonical(r0,c1) +
                       dr*(1-dc)*canonical(r1,c0) +
                       dr*dc*canonical(r1,c1);
        }
    }
    return out;
}

// --- Robust PGM IO ---
Image loadPGM(const string& filename) {
    ifstream file(filename, ios::binary);
    if (!file.is_open()) throw runtime_error("Could not open input file");

    string magic; file >> magic;
    if (magic != "P5") throw runtime_error("Invalid PGM format");

    auto skipComments = [&file]() {
        while (file >> ws && file.peek() == '#') file.ignore(4096, '\n');
    };

    skipComments();
    int w, h, maxVal;
    file >> w >> h;
    skipComments();
    file >> maxVal;
    file.ignore(1);

    Image img; img.width = w; img.height = h;
    img.data.resize(h, w);

    vector<unsigned char> buffer(w * h);
    file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
    
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            img.data(i, j) = static_cast<float>(buffer[i * w + j]);
        }
    }
    return img;
}

void savePGM(const string& filename, const Image& img) {
    ofstream file(filename, ios::binary);
    file << "P5\n" << img.width << " " << img.height << "\n255\n";
    vector<unsigned char> buffer(img.width * img.height);
    for (int i = 0; i < img.height; ++i) {
        for (int j = 0; j < img.width; ++j) {
            buffer[i * img.width + j] = (unsigned char)std::clamp(img.data(i, j), 0.f, 255.f);
        }
    }
    file.write(reinterpret_cast<char*>(buffer.data()), buffer.size());
}

// --- Dictionary Learning ---
MatrixXf learn_dictionary(const Image& img) {
    cout << "[*] Learning Additive Dictionary Atoms..." << endl;
    vector<VectorXf, aligned_allocator<VectorXf>> samples;
    
    int steps[] = {4, 8, 16, 32};
    
    for (int s : steps) {
        for (int i = 0; i + s <= img.height; i += s/2) {
            for (int j = 0; j + s <= img.width; j += s/2) {
                if (rand() % 3 != 0) continue;

                MatrixXf blk = img.data.block(i, j, s, s);
                VectorXf vec = canonicalize(blk);
                
                float mean = vec.mean();
                vec.array() -= mean;
                float norm = vec.norm();
                if (norm > 1e-4) vec /= norm;
                
                samples.push_back(vec);
            }
        }
    }

    if (samples.empty()) throw runtime_error("Image too small");

    MatrixXf dictionary(CANONICAL_DIM, DICT_SIZE);
    mt19937 rng(12345);
    uniform_int_distribution<int> dist(0, samples.size() - 1);
    for (int k = 0; k < DICT_SIZE; ++k) dictionary.col(k) = samples[dist(rng)];

    for (int iter = 0; iter < 15; ++iter) {
        vector<VectorXf, aligned_allocator<VectorXf>> centroids(DICT_SIZE, VectorXf::Zero(CANONICAL_DIM));
        vector<int> counts(DICT_SIZE, 0);

        #pragma omp parallel 
        {
            vector<VectorXf, aligned_allocator<VectorXf>> local_centroids(DICT_SIZE, VectorXf::Zero(CANONICAL_DIM));
            vector<int> local_counts(DICT_SIZE, 0);

            #pragma omp for nowait
            for (size_t v = 0; v < samples.size(); ++v) {
                int best_k;
                (dictionary.transpose() * samples[v]).maxCoeff(&best_k);
                local_centroids[best_k] += samples[v];
                local_counts[best_k]++;
            }

            #pragma omp critical
            for (int k = 0; k < DICT_SIZE; ++k) {
                centroids[k] += local_centroids[k];
                counts[k] += local_counts[k];
            }
        }

        float shift = 0;
        for (int k = 0; k < DICT_SIZE; ++k) {
            if (counts[k] > 0) {
                centroids[k] /= (float)counts[k];
                centroids[k].normalize();
                shift += (centroids[k] - dictionary.col(k)).norm();
                dictionary.col(k) = centroids[k];
            } else {
                dictionary.col(k) = samples[dist(rng)];
            }
        }
        if (shift < 1e-2) break;
    }
    return dictionary;
}

// --- HIERARCHICAL RESIDUAL ENCODING ---
// parent_reconstruction: What the parent level already reconstructed for this region
void process_quadrant(const Eigen::Ref<const MatrixXf>& img_block,
                      const Eigen::Ref<const MatrixXf>& parent_reconstruction,
                      const MatrixXf& dict, 
                      CompressedData& storage,
                      int depth = 0) {    
    int h = img_block.rows();
    int w = img_block.cols();
    
    // 1. Compute RESIDUAL (what parent couldn't capture)
    MatrixXf residual = img_block - parent_reconstruction;
    float residual_variance = residual.array().square().mean();
    
    // 2. Encode THIS level's contribution
    VectorXf vec = canonicalize(residual);
    float beta = vec.mean();
    vec.array() -= beta;
    float alpha = vec.norm();
    if (alpha > 1e-5) vec /= alpha;

    int best_k;
    (dict.transpose() * vec).maxCoeff(&best_k);

    // 3. Reconstruct what THIS level adds
    VectorXf atom = dict.col(best_k);
    atom = (atom * alpha).array() + beta;
    MatrixXf my_contribution = reconstruction_map(atom, h, w);
    
    // 4. Total reconstruction so far = parent + my_contribution
    MatrixXf current_reconstruction = parent_reconstruction + my_contribution;
    
    // 5. Calculate remaining error
    MatrixXf final_residual = img_block - current_reconstruction;
    float final_variance = final_residual.array().square().mean();
    
    // 6. Decision: Split or stop?
    bool can_split = (h >= 2*MIN_BLOCK && w >= 2*MIN_BLOCK);
    bool must_split = (h > MAX_BLOCK || w > MAX_BLOCK);
    bool want_split = final_variance > RESIDUAL_THRESHOLD;
    bool should_split = must_split || (can_split && want_split);

    // 7. Store this node's encoding
    storage.split_tree.push_back(should_split);
    storage.indices.push_back((uint8_t)best_k);
    
    // Quantize with sign preservation for residuals
    int8_t scale_q = (int8_t)std::clamp(alpha * 0.5f, -127.f, 127.f);
    int8_t offset_q = (int8_t)std::clamp(beta, -127.f, 127.f);
    storage.scales.push_back(scale_q);
    storage.offsets.push_back(offset_q);

    if (should_split) {
        // Recursively refine with residuals
        int half_h = h / 2;
        int half_w = w / 2;
        
        // Pass current_reconstruction as the new parent baseline
        process_quadrant(img_block.block(0, 0, half_h, half_w),
                        current_reconstruction.block(0, 0, half_h, half_w),
                        dict, storage, depth+1);
        
        process_quadrant(img_block.block(0, half_w, half_h, w - half_w),
                        current_reconstruction.block(0, half_w, half_h, w - half_w),
                        dict, storage, depth+1);
        
        process_quadrant(img_block.block(half_h, 0, h - half_h, half_w),
                        current_reconstruction.block(half_h, 0, h - half_h, half_w),
                        dict, storage, depth+1);
        
        process_quadrant(img_block.block(half_h, half_w, h - half_h, w - half_w),
                        current_reconstruction.block(half_h, half_w, h - half_h, w - half_w),
                        dict, storage, depth+1);
    }
}

void compress(const string& inputFile, const string& outputFile) {
    Image img = loadPGM(inputFile);
    
    // Pad to power of 2
    int padded_h = 1, padded_w = 1;
    while (padded_h < img.height) padded_h *= 2;
    while (padded_w < img.width) padded_w *= 2;
    
    if (padded_h != img.height || padded_w != img.width) {
        MatrixXf padded = MatrixXf::Zero(padded_h, padded_w);
        padded.block(0, 0, img.height, img.width) = img.data;
        img.data = padded;
        img.height = padded_h;
        img.width = padded_w;
        cout << "[*] Padded to " << padded_w << "x" << padded_h << endl;
    }

    MatrixXf dictionary = learn_dictionary(img);
    CompressedData data;
    data.width = img.width;
    data.height = img.height;
    data.dictionary = dictionary;

    cout << "[*] Hierarchical Residual Encoding (algebraic summation)..." << endl;
    
    // Start with zero baseline (no parent)
    MatrixXf zero_baseline = MatrixXf::Zero(img.height, img.width);
    process_quadrant(img.data, zero_baseline, dictionary, data);

    // Serialization
    ofstream outFile(outputFile, ios::binary);
    boost::iostreams::filtering_streambuf<boost::iostreams::output> out;
    out.push(boost::iostreams::zlib_compressor());
    out.push(outFile);
    ostream os(&out);

    os.write((char*)&data.width, sizeof(int));
    os.write((char*)&data.height, sizeof(int));
    os.write((char*)data.dictionary.data(), data.dictionary.size() * sizeof(float));

    int tree_size = data.split_tree.size();
    os.write((char*)&tree_size, sizeof(int));
    
    vector<uint8_t> packed_tree((tree_size + 7) / 8, 0);
    for(int i=0; i<tree_size; ++i) {
        if(data.split_tree[i]) packed_tree[i/8] |= (1 << (i%8));
    }
    os.write((char*)packed_tree.data(), packed_tree.size());

    // All nodes store data (not just leaves!)
    os.write((char*)data.indices.data(), tree_size);
    os.write((char*)data.scales.data(), tree_size);
    os.write((char*)data.offsets.data(), tree_size);

    cout << "[*] Compressed. Total nodes: " << tree_size 
         << " (hierarchical atoms)" << endl;
}

// --- HIERARCHICAL RECONSTRUCTION ---
struct StreamReader {
    vector<bool> split_tree;
    vector<uint8_t> indices;
    vector<int8_t> scales;
    vector<int8_t> offsets;
    int idx = 0;
};

void reconstruct_quadrant(Eigen::Ref<MatrixXf> img_block,
                         const Eigen::Ref<const MatrixXf>& parent_reconstruction,
                         const MatrixXf& dict, 
                         StreamReader& sr) {
    if (sr.idx >= (int)sr.split_tree.size()) return;
    
    int h = img_block.rows();
    int w = img_block.cols();
    
    // 1. Read this node's atom
    bool is_split = sr.split_tree[sr.idx];
    uint8_t atom_idx = sr.indices[sr.idx];
    int8_t sc = sr.scales[sr.idx];
    int8_t off = sr.offsets[sr.idx];
    sr.idx++;
    
    // 2. Decode this level's contribution
    float alpha = (float)sc * 2.0f;
    float beta = (float)off;
    
    VectorXf atom = dict.col(atom_idx);
    atom = (atom * alpha).array() + beta;
    MatrixXf my_contribution = reconstruction_map(atom, h, w);
    
    // 3. Algebraic sum: parent + this_level
    MatrixXf current_reconstruction = parent_reconstruction + my_contribution;
    
    if (is_split) {
        // 4. Recursively reconstruct children with accumulated baseline
        int half_h = h / 2;
        int half_w = w / 2;
        
        reconstruct_quadrant(img_block.block(0, 0, half_h, half_w),
                           current_reconstruction.block(0, 0, half_h, half_w),
                           dict, sr);
        
        reconstruct_quadrant(img_block.block(0, half_w, half_h, w - half_w),
                           current_reconstruction.block(0, half_w, half_h, w - half_w),
                           dict, sr);
        
        reconstruct_quadrant(img_block.block(half_h, 0, h - half_h, half_w),
                           current_reconstruction.block(half_h, 0, h - half_h, half_w),
                           dict, sr);
        
        reconstruct_quadrant(img_block.block(half_h, half_w, h - half_h, w - half_w),
                           current_reconstruction.block(half_h, half_w, h - half_h, w - half_w),
                           dict, sr);
    } else {
        // 5. Leaf: store the final sum
        img_block = current_reconstruction;
    }
}

void decompress(const string& inputFile, const string& outputFile) {
    ifstream inFile(inputFile, ios::binary);
    if (!inFile) throw runtime_error("Cannot open file");

    boost::iostreams::filtering_streambuf<boost::iostreams::input> in;
    in.push(boost::iostreams::zlib_decompressor());
    in.push(inFile);
    istream is(&in);

    int w, h;
    is.read((char*)&w, sizeof(int));
    is.read((char*)&h, sizeof(int));

    MatrixXf dict(CANONICAL_DIM, DICT_SIZE);
    is.read((char*)dict.data(), dict.size() * sizeof(float));

    int tree_size;
    is.read((char*)&tree_size, sizeof(int));
    vector<uint8_t> packed_tree((tree_size + 7) / 8);
    is.read((char*)packed_tree.data(), packed_tree.size());

    StreamReader sr;
    sr.split_tree.resize(tree_size);
    for(int i=0; i<tree_size; ++i) {
        sr.split_tree[i] = (packed_tree[i/8] >> (i%8)) & 1;
    }

    sr.indices.resize(tree_size);
    sr.scales.resize(tree_size);
    sr.offsets.resize(tree_size);

    is.read((char*)sr.indices.data(), tree_size);
    is.read((char*)sr.scales.data(), tree_size);
    is.read((char*)sr.offsets.data(), tree_size);

    Image img; 
    img.width = w; 
    img.height = h;
    img.data.resize(h, w);

    cout << "[*] Reconstructing via algebraic summation..." << endl;
    
    MatrixXf zero_baseline = MatrixXf::Zero(h, w);
    reconstruct_quadrant(img.data, zero_baseline, dict, sr);

    savePGM(outputFile, img);
    cout << "[*] Decompression complete." << endl;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        cout << "Usage: " << argv[0] << " <c/d> <input> <output>\n";
        cout << "  c - compress PGM image\n";
        cout << "  d - decompress to PGM\n";
        return 1;
    }
    
    try {
        string mode = argv[1];
        if (mode == "c") compress(argv[2], argv[3]);
        else if (mode == "d") decompress(argv[2], argv[3]);
        else cout << "Invalid mode. Use 'c' or 'd'\n";
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}
