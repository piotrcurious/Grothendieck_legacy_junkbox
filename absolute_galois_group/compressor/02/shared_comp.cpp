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
 * Topological Algebraic Compressor (TAC)
 * * Features:
 * 1. Quadtree Decomposition: Adapts to image entropy (Variable Block Size).
 * 2. Scale-Invariant Homology: Maps N*N blocks to a canonical 8x8 manifold.
 * 3. Structure-of-Arrays Indexing: Separated streams for higher Zlib ratios.
 * * Compile:
 * g++ -O3 -I /usr/include/eigen3 tac_compressor.cpp -o tac -lboost_iostreams -lz -fopenmp
 */

using namespace Eigen;
using namespace std;

// --- Tuning Parameters ---
const int CANONICAL_SIZE = 8;               // The "Shape Space" dimension
const int CANONICAL_DIM = CANONICAL_SIZE * CANONICAL_SIZE;
const int DICT_SIZE = 256; 
const int MIN_BLOCK = 4;
const int MAX_BLOCK = 32;
const float VAR_THRESHOLD = 15.0f;          // Variance threshold to trigger split

struct Image {
    int width, height;
    MatrixXf data;
};

// Separated streams for better entropy coding
struct CompressedData {
    int width, height;
    MatrixXf dictionary;
    std::vector<bool> split_tree;     // The quadtree structure
    std::vector<uint8_t> indices;     // Dictionary atoms
    std::vector<uint8_t> scales;      // Contrast
    std::vector<uint8_t> offsets;     // Brightness
};

// --- Helper: Resampling ---
// Maps a variable size block to the canonical size (Downsampling)
VectorXf canonicalize(const MatrixXf& block) {
    if (block.rows() == CANONICAL_SIZE && block.cols() == CANONICAL_SIZE) {
        return Map<const VectorXf>(block.data(), CANONICAL_DIM);
    }
    
    // Simple block averaging resize
    MatrixXf resized(CANONICAL_SIZE, CANONICAL_SIZE);
    float r_step = (float)block.rows() / CANONICAL_SIZE;
    float c_step = (float)block.cols() / CANONICAL_SIZE;

    for (int i = 0; i < CANONICAL_SIZE; ++i) {
        for (int j = 0; j < CANONICAL_SIZE; ++j) {
            // Sample center of the target region
            int r = std::min((int)(i * r_step), (int)block.rows() - 1);
            int c = std::min((int)(j * c_step), (int)block.cols() - 1);
            resized(i, j) = block(r, c);
        }
    }
    return Map<VectorXf>(resized.data(), CANONICAL_DIM);
}

// Maps a canonical vector back to variable size block (Upsampling)
MatrixXf reconstruction_map(const VectorXf& vec, int target_rows, int target_cols) {
    MatrixXf canonical = Map<const MatrixXf>(vec.data(), CANONICAL_SIZE, CANONICAL_SIZE);
    
    if (target_rows == CANONICAL_SIZE && target_cols == CANONICAL_SIZE) {
        return canonical;
    }

    MatrixXf out(target_rows, target_cols);
    float r_ratio = (float)(CANONICAL_SIZE - 1) / (std::max(1, target_rows - 1));
    float c_ratio = (float)(CANONICAL_SIZE - 1) / (std::max(1, target_cols - 1));

    for (int i = 0; i < target_rows; ++i) {
        for (int j = 0; j < target_cols; ++j) {
            // Bilinear interpolation (simplified to nearest for speed/demo)
            // Ideally, replace with proper interpolation for fewer artifacts
            int r = (int)(i * r_ratio);
            int c = (int)(j * c_ratio);
            out(i, j) = canonical(r, c);
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
    cout << "[*] Learning Scale-Invariant Cohomologies..." << endl;
    vector<VectorXf, aligned_allocator<VectorXf>> samples;
    
    // Multi-scale sampling: grab blocks of various sizes and canonicalize them
    int steps[] = {4, 8, 16};
    
    for (int s : steps) {
        for (int i = 0; i + s <= img.height; i += s) {
            for (int j = 0; j + s <= img.width; j += s) {
                // Stochastic skipping to speed up training
                if (rand() % 2 != 0) continue; 

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

    // Standard K-Means
    MatrixXf dictionary(CANONICAL_DIM, DICT_SIZE);
    mt19937 rng(12345);
    uniform_int_distribution<int> dist(0, samples.size() - 1);
    for (int k = 0; k < DICT_SIZE; ++k) dictionary.col(k) = samples[dist(rng)];

    for (int iter = 0; iter < 10; ++iter) {
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

// --- Recursive Quadtree Compression ---
void process_quadrant(const Eigen::Ref<const MatrixXf>& img_block, 
                      const MatrixXf& dict, 
                      CompressedData& storage) {    
    int h = img_block.rows();
    int w = img_block.cols();
    
    // 1. Calculate Variance (Energy)
    float mean = img_block.mean();
    float variance = (img_block.array() - mean).square().sum() / (w * h);

    // 2. Decision: Split or Leaf?
    // Split if variance is high AND we are not at minimum block size
    // Also force split if we are larger than MAX_BLOCK
    bool should_split = (variance > VAR_THRESHOLD && (h > MIN_BLOCK && w > MIN_BLOCK)) 
                        || (h > MAX_BLOCK || w > MAX_BLOCK);

    // Write split bit
    storage.split_tree.push_back(should_split);

    if (should_split) {
        int half_h = h / 2;
        int half_w = w / 2;
        // TL, TR, BL, BR
        process_quadrant(img_block.block(0, 0, half_h, half_w), dict, storage);
        process_quadrant(img_block.block(0, half_w, half_h, w - half_w), dict, storage);
        process_quadrant(img_block.block(half_h, 0, h - half_h, half_w), dict, storage);
        process_quadrant(img_block.block(half_h, half_w, h - half_h, w - half_w), dict, storage);
    } else {
        // --- Leaf Node Encoding ---
        // 1. Canonicalize (Map to Shape Space)
        VectorXf vec = canonicalize(img_block);

        // 2. Normalize
        float beta = vec.mean();
        vec.array() -= beta;
        float alpha = vec.norm();
        if (alpha > 1e-5) vec /= alpha;

        // 3. Find best Atom
        int best_k;
        (dict.transpose() * vec).maxCoeff(&best_k);

        // 4. Quantize and Store (SoA Layout)
        storage.indices.push_back((uint8_t)best_k);
        storage.scales.push_back((uint8_t)std::min(255.f, std::max(0.f, alpha * 0.5f)));
        storage.offsets.push_back((uint8_t)std::min(255.f, std::max(0.f, beta)));
    }
}

void compress(const string& inputFile, const string& outputFile) {
    Image img = loadPGM(inputFile);
    
    // Ensure dimensions are divisible by min block for simplicity in this demo
    // Real impl would pad
    if (img.width % MIN_BLOCK != 0 || img.height % MIN_BLOCK != 0) {
        cerr << "Warning: Resizing image to be divisible by " << MIN_BLOCK << endl;
        img.width = (img.width / MIN_BLOCK) * MIN_BLOCK;
        img.height = (img.height / MIN_BLOCK) * MIN_BLOCK;
        img.data = img.data.block(0,0,img.height, img.width).eval();
    }

    MatrixXf dictionary = learn_dictionary(img);
    CompressedData data;
    data.width = img.width;
    data.height = img.height;
    data.dictionary = dictionary;

    cout << "[*] Quadtree Decomposition & Encoding..." << endl;
    process_quadrant(img.data, dictionary, data);

    // --- Serialization ---
    ofstream outFile(outputFile, ios::binary);
    boost::iostreams::filtering_streambuf<boost::iostreams::output> out;
    out.push(boost::iostreams::zlib_compressor());
    out.push(outFile);
    ostream os(&out);

    // Write Header
    os.write((char*)&data.width, sizeof(int));
    os.write((char*)&data.height, sizeof(int));
    os.write((char*)data.dictionary.data(), data.dictionary.size() * sizeof(float));

    // Write Structure (Bit Packing for Bool vector)
    int tree_size = data.split_tree.size();
    os.write((char*)&tree_size, sizeof(int));
    
    // Pack bools into bytes
    vector<uint8_t> packed_tree((tree_size + 7) / 8, 0);
    for(int i=0; i<tree_size; ++i) {
        if(data.split_tree[i]) packed_tree[i/8] |= (1 << (i%8));
    }
    os.write((char*)packed_tree.data(), packed_tree.size());

    // Write Data Streams (Contiguous arrays compress better)
    os.write((char*)data.indices.data(), data.indices.size());
    os.write((char*)data.scales.data(), data.scales.size());
    os.write((char*)data.offsets.data(), data.offsets.size());

    cout << "[*] Compressed. Nodes: " << tree_size << " Leaves: " << data.indices.size() << endl;
}

// --- Decompression Helpers ---
struct StreamReader {
    vector<bool> split_tree;
    vector<uint8_t> indices;
    vector<uint8_t> scales;
    vector<uint8_t> offsets;
    int tree_idx = 0, leaf_idx = 0;
};

// Eigen::Ref erases the "Block" type, preventing infinite template recursion
void reconstruct_quadrant(Eigen::Ref<MatrixXf> img_block, const MatrixXf& dict, StreamReader& sr) {
    if (sr.tree_idx >= (int)sr.split_tree.size()) return;
    
    bool is_split = sr.split_tree[sr.tree_idx++];
    
    int h = img_block.rows();
    int w = img_block.cols();

    if (is_split) {
        int half_h = h / 2;
        int half_w = w / 2;
        
        // Split into 4 quadrants
        // Eigen::Ref handles these sub-blocks automatically
        reconstruct_quadrant(img_block.block(0, 0, half_h, half_w), dict, sr);
        reconstruct_quadrant(img_block.block(0, half_w, half_h, w - half_w), dict, sr);
        reconstruct_quadrant(img_block.block(half_h, 0, h - half_h, half_w), dict, sr);
        reconstruct_quadrant(img_block.block(half_h, half_w, h - half_h, w - half_w), dict, sr);
    } else {
        // Leaf - Reconstruct
        if (sr.leaf_idx >= (int)sr.indices.size()) return;
        
        uint8_t idx = sr.indices[sr.leaf_idx];
        uint8_t sc = sr.scales[sr.leaf_idx];
        uint8_t off = sr.offsets[sr.leaf_idx];
        sr.leaf_idx++;

        float alpha = (float)sc * 2.0f;
        float beta = (float)off;

        VectorXf vec = dict.col(idx);
        vec = (vec * alpha).array() + beta;

        // Apply the reconstruction back to this block's memory
        img_block = reconstruction_map(vec, h, w);
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

    // Determine number of leaves (count 0s in tree)
    int leaves = 0;
    for(bool b : sr.split_tree) if(!b) leaves++;

    sr.indices.resize(leaves);
    sr.scales.resize(leaves);
    sr.offsets.resize(leaves);

    is.read((char*)sr.indices.data(), leaves);
    is.read((char*)sr.scales.data(), leaves);
    is.read((char*)sr.offsets.data(), leaves);

    Image img; img.width = w; img.height = h;
    img.data.resize(h, w);

    cout << "[*] Reconstructing Variable Structures..." << endl;
    reconstruct_quadrant(img.data, dict, sr);

    savePGM(outputFile, img);
    cout << "[*] Done." << endl;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        cout << "Usage: " << argv[0] << " <c/d> <input> <output>\n";
        return 1;
    }
    string mode = argv[1];
    if (mode == "c") compress(argv[2], argv[3]);
    else if (mode == "d") decompress(argv[2], argv[3]);
    return 0;
}
