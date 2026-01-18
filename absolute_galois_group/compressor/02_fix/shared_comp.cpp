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
 * Topological Algebraic Compressor (TAC) - FIXED VERSION
 * Features:
 * 1. TRUE Adaptive Quadtree Decomposition based on variance
 * 2. Scale-Invariant Homology: Maps N*N blocks to canonical 8x8 manifold
 * 3. Structure-of-Arrays Indexing for better compression
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
const float VAR_THRESHOLD = 200.0f;  // Higher = less splitting (more compression)

struct Image {
    int width, height;
    MatrixXf data;
};

struct CompressedData {
    int width, height;
    MatrixXf dictionary;
    std::vector<bool> split_tree;
    std::vector<uint8_t> indices;
    std::vector<uint8_t> scales;
    std::vector<uint8_t> offsets;
    std::vector<uint16_t> block_sizes;  // NEW: Track actual block dimensions
};

// --- Helper: Resampling with Bilinear Interpolation ---
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
            
            // Bilinear interpolation
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

// --- Scale-Invariant Dictionary Learning ---
MatrixXf learn_dictionary(const Image& img) {
    cout << "[*] Learning Scale-Invariant Dictionary..." << endl;
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

    // K-Means Clustering
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

// --- FIXED Recursive Quadtree Compression ---
void process_quadrant(const Eigen::Ref<const MatrixXf>& img_block, 
                      const MatrixXf& dict, 
                      CompressedData& storage) {    
    int h = img_block.rows();
    int w = img_block.cols();
    
    // Calculate variance
    float mean = img_block.mean();
    float variance = (img_block.array() - mean).square().mean();

    // FIXED: Proper split decision logic
    bool can_split = (h >= 2*MIN_BLOCK && w >= 2*MIN_BLOCK);
    bool must_split = (h > MAX_BLOCK || w > MAX_BLOCK);
    bool should_split = must_split || (can_split && variance > VAR_THRESHOLD);

    storage.split_tree.push_back(should_split);

    if (should_split) {
        // Recursively split into 4 quadrants
        int half_h = h / 2;
        int half_w = w / 2;
        
        process_quadrant(img_block.block(0, 0, half_h, half_w), dict, storage);
        process_quadrant(img_block.block(0, half_w, half_h, w - half_w), dict, storage);
        process_quadrant(img_block.block(half_h, 0, h - half_h, half_w), dict, storage);
        process_quadrant(img_block.block(half_h, half_w, h - half_h, w - half_w), dict, storage);
    } else {
        // LEAF NODE - Encode this block
        VectorXf vec = canonicalize(img_block);

        float beta = vec.mean();
        vec.array() -= beta;
        float alpha = vec.norm();
        if (alpha > 1e-5) vec /= alpha;

        int best_k;
        (dict.transpose() * vec).maxCoeff(&best_k);

        // Store leaf data
        storage.indices.push_back((uint8_t)best_k);
        storage.scales.push_back((uint8_t)std::min(255.f, std::max(0.f, alpha * 0.5f)));
        storage.offsets.push_back((uint8_t)std::min(255.f, std::max(0.f, beta)));
        storage.block_sizes.push_back((uint16_t)((h << 8) | w));  // Pack dimensions
    }
}

void compress(const string& inputFile, const string& outputFile) {
    Image img = loadPGM(inputFile);
    
    // Pad image to be power of 2 for cleaner quadtree
    int padded_h = 1;
    int padded_w = 1;
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

    cout << "[*] Adaptive Quadtree Decomposition (threshold=" << VAR_THRESHOLD << ")..." << endl;
    process_quadrant(img.data, dictionary, data);

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

    int leaf_count = data.indices.size();
    os.write((char*)&leaf_count, sizeof(int));
    
    os.write((char*)data.indices.data(), leaf_count);
    os.write((char*)data.scales.data(), leaf_count);
    os.write((char*)data.offsets.data(), leaf_count);
    os.write((char*)data.block_sizes.data(), leaf_count * sizeof(uint16_t));

    cout << "[*] Compressed. Tree nodes: " << tree_size 
         << " | Leaf blocks: " << leaf_count << endl;
}

// --- FIXED Decompression ---
struct StreamReader {
    vector<bool> split_tree;
    vector<uint8_t> indices;
    vector<uint8_t> scales;
    vector<uint8_t> offsets;
    vector<uint16_t> block_sizes;
    int tree_idx = 0, leaf_idx = 0;
};

void reconstruct_quadrant(Eigen::Ref<MatrixXf> img_block, 
                         const MatrixXf& dict, 
                         StreamReader& sr) {
    if (sr.tree_idx >= (int)sr.split_tree.size()) return;
    
    bool is_split = sr.split_tree[sr.tree_idx++];
    
    int h = img_block.rows();
    int w = img_block.cols();

    if (is_split) {
        int half_h = h / 2;
        int half_w = w / 2;
        
        reconstruct_quadrant(img_block.block(0, 0, half_h, half_w), dict, sr);
        reconstruct_quadrant(img_block.block(0, half_w, half_h, w - half_w), dict, sr);
        reconstruct_quadrant(img_block.block(half_h, 0, h - half_h, half_w), dict, sr);
        reconstruct_quadrant(img_block.block(half_h, half_w, h - half_h, w - half_w), dict, sr);
    } else {
        if (sr.leaf_idx >= (int)sr.indices.size()) return;
        
        uint8_t idx = sr.indices[sr.leaf_idx];
        uint8_t sc = sr.scales[sr.leaf_idx];
        uint8_t off = sr.offsets[sr.leaf_idx];
        uint16_t packed_size = sr.block_sizes[sr.leaf_idx];
        sr.leaf_idx++;

        // Unpack dimensions
        int block_h = (packed_size >> 8) & 0xFF;
        int block_w = packed_size & 0xFF;

        float alpha = (float)sc * 2.0f;
        float beta = (float)off;

        VectorXf vec = dict.col(idx);
        vec = (vec * alpha).array() + beta;

        // Reconstruct to actual block size
        img_block = reconstruction_map(vec, block_h, block_w);
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

    int leaf_count;
    is.read((char*)&leaf_count, sizeof(int));

    sr.indices.resize(leaf_count);
    sr.scales.resize(leaf_count);
    sr.offsets.resize(leaf_count);
    sr.block_sizes.resize(leaf_count);

    is.read((char*)sr.indices.data(), leaf_count);
    is.read((char*)sr.scales.data(), leaf_count);
    is.read((char*)sr.offsets.data(), leaf_count);
    is.read((char*)sr.block_sizes.data(), leaf_count * sizeof(uint16_t));

    Image img; 
    img.width = w; 
    img.height = h;
    img.data.resize(h, w);

    cout << "[*] Reconstructing from " << leaf_count << " variable-size blocks..." << endl;
    reconstruct_quadrant(img.data, dict, sr);

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
