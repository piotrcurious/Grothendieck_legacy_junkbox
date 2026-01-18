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

using namespace Eigen;
using namespace std;

// --- Tuning Parameters ---
const int CANONICAL_SIZE = 8;
const int CANONICAL_DIM = CANONICAL_SIZE * CANONICAL_SIZE;
const int DICT_SIZE = 256; 
const int MIN_BLOCK = 4;
const int MAX_BLOCK = 64;
const float VAR_THRESHOLD = 200.0f; 

struct Image {
    int width, height;
    MatrixXf data;
};

struct CompressedData {
    int orig_width, orig_height; // Added to track original size
    int padded_width, padded_height;
    MatrixXf dictionary;
    std::vector<bool> split_tree;
    std::vector<uint8_t> indices;
    std::vector<uint8_t> scales;
    std::vector<uint8_t> offsets;
    std::vector<uint16_t> block_sizes;
};

// --- Helper Functions (canonicalize and reconstruction_map remain the same) ---
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
            resized(i, j) = (1-dr)*(1-dc)*block(r0,c0) + (1-dr)*dc*block(r0,c1) + dr*(1-dc)*block(r1,c0) + dr*dc*block(r1,c1);
        }
    }
    return Map<VectorXf>(resized.data(), CANONICAL_DIM);
}

MatrixXf reconstruction_map(const VectorXf& vec, int target_rows, int target_cols) {
    MatrixXf canonical = Map<const MatrixXf>(vec.data(), CANONICAL_SIZE, CANONICAL_SIZE);
    if (target_rows == CANONICAL_SIZE && target_cols == CANONICAL_SIZE) return canonical;
    MatrixXf out(target_rows, target_cols);
    float r_ratio = (float)(CANONICAL_SIZE - 1) / std::max(1, target_rows - 1);
    float c_ratio = (float)(CANONICAL_SIZE - 1) / std::max(1, target_cols - 1);
    for (int i = 0; i < target_rows; ++i) {
        for (int j = 0; j < target_cols; ++j) {
            float src_r = i * r_ratio; float src_c = j * c_ratio;
            int r0 = (int)src_r; int c0 = (int)src_c;
            int r1 = std::min(r0 + 1, CANONICAL_SIZE - 1);
            int c1 = std::min(c0 + 1, CANONICAL_SIZE - 1);
            float dr = src_r - r0; float dc = src_c - c0;
            out(i, j) = (1-dr)*(1-dc)*canonical(r0,c0) + (1-dr)*dc*canonical(r0,c1) + dr*(1-dc)*canonical(r1,c0) + dr*dc*canonical(r1,c1);
        }
    }
    return out;
}

// --- PGM IO (Logic remains the same) ---
Image loadPGM(const string& filename) {
    ifstream file(filename, ios::binary);
    if (!file.is_open()) throw runtime_error("Could not open input file");
    string magic; file >> magic;
    if (magic != "P5") throw runtime_error("Invalid PGM format");
    auto skipComments = [&file]() { while (file >> ws && file.peek() == '#') file.ignore(4096, '\n'); };
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
    for (int i = 0; i < h; ++i) for (int j = 0; j < w; ++j) img.data(i, j) = static_cast<float>(buffer[i * w + j]);
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

    if (samples.empty()) throw runtime_error("Image too small for dictionary learning");

    MatrixXf dictionary(CANONICAL_DIM, DICT_SIZE);
    mt19937 rng(12345);
    uniform_int_distribution<int> dist(0, (int)samples.size() - 1);
    for (int k = 0; k < DICT_SIZE; ++k) dictionary.col(k) = samples[dist(rng)];

    for (int iter = 0; iter < 15; ++iter) {
        vector<VectorXf, aligned_allocator<VectorXf>> centroids(DICT_SIZE, VectorXf::Zero(CANONICAL_DIM));
        vector<int> counts(DICT_SIZE, 0);

        // FIXED: The brace must be on a new line after the pragma
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
            {
                for (int k = 0; k < DICT_SIZE; ++k) {
                    centroids[k] += local_centroids[k];
                    counts[k] += local_counts[k];
                }
            }
        }

        float shift = 0;
        for (int k = 0; k < DICT_SIZE; ++k) {
            if (counts[k] > 0) {
                centroids[k] /= (float)counts[k];
                centroids[k].normalize();
                shift += (centroids[k] - dictionary.col(k)).norm();
                dictionary.col(k) = centroids[k];
            }
        }
        if (shift < 1e-2) break;
    }
    return dictionary;
}

void process_quadrant(const Eigen::Ref<const MatrixXf>& img_block, const MatrixXf& dict, CompressedData& storage) {    
    int h = img_block.rows(); int w = img_block.cols();
    float mean = img_block.mean();
    float variance = (img_block.array() - mean).square().mean();
    bool can_split = (h >= 2*MIN_BLOCK && w >= 2*MIN_BLOCK);
    bool must_split = (h > MAX_BLOCK || w > MAX_BLOCK);
    bool should_split = must_split || (can_split && variance > VAR_THRESHOLD);
    storage.split_tree.push_back(should_split);
    if (should_split) {
        int half_h = h / 2; int half_w = w / 2;
        process_quadrant(img_block.block(0, 0, half_h, half_w), dict, storage);
        process_quadrant(img_block.block(0, half_w, half_h, w - half_w), dict, storage);
        process_quadrant(img_block.block(half_h, 0, h - half_h, half_w), dict, storage);
        process_quadrant(img_block.block(half_h, half_w, h - half_h, w - half_w), dict, storage);
    } else {
        VectorXf vec = canonicalize(img_block);
        float beta = vec.mean(); vec.array() -= beta;
        float alpha = vec.norm(); if (alpha > 1e-5) vec /= alpha;
        int best_k; (dict.transpose() * vec).maxCoeff(&best_k);
        storage.indices.push_back((uint8_t)best_k);
        storage.scales.push_back((uint8_t)std::min(255.f, std::max(0.f, alpha * 0.5f)));
        storage.offsets.push_back((uint8_t)std::min(255.f, std::max(0.f, beta)));
        storage.block_sizes.push_back((uint16_t)((h << 8) | w));
    }
}

// --- FIXED COMPRESS ---
void compress(const string& inputFile, const string& outputFile) {
    Image img = loadPGM(inputFile);
    
    // 1. Store original dimensions
    int orig_h = img.height;
    int orig_w = img.width;

    // 2. Determine padded dimensions
    int padded_h = 1;
    int padded_w = 1;
    while (padded_h < orig_h) padded_h *= 2;
    while (padded_w < orig_w) padded_w *= 2;
    
    // Create padded version for processing
    MatrixXf processed_data = MatrixXf::Zero(padded_h, padded_w);
    processed_data.block(0, 0, orig_h, orig_w) = img.data;

    MatrixXf dictionary = learn_dictionary(img);
    CompressedData data;
    data.orig_width = orig_w;   // NEW
    data.orig_height = orig_h;  // NEW
    data.padded_width = padded_w;
    data.padded_height = padded_h;
    data.dictionary = dictionary;

    process_quadrant(processed_data, dictionary, data);

    ofstream outFile(outputFile, ios::binary);
    boost::iostreams::filtering_streambuf<boost::iostreams::output> out;
    out.push(boost::iostreams::zlib_compressor());
    out.push(outFile);
    ostream os(&out);

    // 3. Write BOTH original and padded sizes
    os.write((char*)&data.orig_width, sizeof(int));
    os.write((char*)&data.orig_height, sizeof(int));
    os.write((char*)&data.padded_width, sizeof(int));
    os.write((char*)&data.padded_height, sizeof(int));
    
    os.write((char*)data.dictionary.data(), data.dictionary.size() * sizeof(float));
    int tree_size = data.split_tree.size();
    os.write((char*)&tree_size, sizeof(int));
    vector<uint8_t> packed_tree((tree_size + 7) / 8, 0);
    for(int i=0; i<tree_size; ++i) if(data.split_tree[i]) packed_tree[i/8] |= (1 << (i%8));
    os.write((char*)packed_tree.data(), packed_tree.size());
    int leaf_count = data.indices.size();
    os.write((char*)&leaf_count, sizeof(int));
    os.write((char*)data.indices.data(), leaf_count);
    os.write((char*)data.scales.data(), leaf_count);
    os.write((char*)data.offsets.data(), leaf_count);
    os.write((char*)data.block_sizes.data(), leaf_count * sizeof(uint16_t));
}

// --- FIXED DECOMPRESS ---
struct StreamReader {
    vector<bool> split_tree;
    vector<uint8_t> indices;
    vector<uint8_t> scales;
    vector<uint8_t> offsets;
    vector<uint16_t> block_sizes;
    int tree_idx = 0, leaf_idx = 0;
};

void reconstruct_quadrant(Eigen::Ref<MatrixXf> img_block, const MatrixXf& dict, StreamReader& sr) {
    if (sr.tree_idx >= (int)sr.split_tree.size()) return;
    bool is_split = sr.split_tree[sr.tree_idx++];
    int h = img_block.rows(); int w = img_block.cols();
    if (is_split) {
        int half_h = h / 2; int half_w = w / 2;
        reconstruct_quadrant(img_block.block(0, 0, half_h, half_w), dict, sr);
        reconstruct_quadrant(img_block.block(0, half_w, half_h, w - half_w), dict, sr);
        reconstruct_quadrant(img_block.block(half_h, 0, h - half_h, half_w), dict, sr);
        reconstruct_quadrant(img_block.block(half_h, half_w, h - half_h, w - half_w), dict, sr);
    } else {
        if (sr.leaf_idx >= (int)sr.indices.size()) return;
        uint8_t idx = sr.indices[sr.leaf_idx];
        uint8_t sc = sr.scales[sr.leaf_idx];
        uint8_t off = sr.offsets[sr.leaf_idx];
        uint16_t packed_size = sr.block_sizes[sr.leaf_idx++];
        int block_h = (packed_size >> 8) & 0xFF;
        int block_w = packed_size & 0xFF;
        float alpha = (float)sc * 2.0f; float beta = (float)off;
        VectorXf vec = (dict.col(idx) * alpha).array() + beta;
        img_block = reconstruction_map(vec, block_h, block_w);
    }
}

void decompress(const string& inputFile, const string& outputFile) {
    ifstream inFile(inputFile, ios::binary);
    boost::iostreams::filtering_streambuf<boost::iostreams::input> in;
    in.push(boost::iostreams::zlib_decompressor());
    in.push(inFile);
    istream is(&in);

    int orig_w, orig_h, padded_w, padded_h;
    is.read((char*)&orig_w, sizeof(int));
    is.read((char*)&orig_h, sizeof(int));
    is.read((char*)&padded_w, sizeof(int));
    is.read((char*)&padded_h, sizeof(int));

    MatrixXf dict(CANONICAL_DIM, DICT_SIZE);
    is.read((char*)dict.data(), dict.size() * sizeof(float));
    int tree_size; is.read((char*)&tree_size, sizeof(int));
    vector<uint8_t> packed_tree((tree_size + 7) / 8);
    is.read((char*)packed_tree.data(), packed_tree.size());
    StreamReader sr;
    sr.split_tree.resize(tree_size);
    for(int i=0; i<tree_size; ++i) sr.split_tree[i] = (packed_tree[i/8] >> (i%8)) & 1;
    int leaf_count; is.read((char*)&leaf_count, sizeof(int));
    sr.indices.resize(leaf_count); sr.scales.resize(leaf_count);
    sr.offsets.resize(leaf_count); sr.block_sizes.resize(leaf_count);
    is.read((char*)sr.indices.data(), leaf_count);
    is.read((char*)sr.scales.data(), leaf_count);
    is.read((char*)sr.offsets.data(), leaf_count);
    is.read((char*)sr.block_sizes.data(), leaf_count * sizeof(uint16_t));

    // 1. Reconstruct into the PADDED size
    MatrixXf full_padded = MatrixXf::Zero(padded_h, padded_w);
    reconstruct_quadrant(full_padded, dict, sr);

    // 2. Crop to the ORIGINAL size
    Image img;
    img.width = orig_w;
    img.height = orig_h;
    img.data = full_padded.block(0, 0, orig_h, orig_w);

    savePGM(outputFile, img);
}

int main(int argc, char* argv[]) {
    if (argc < 4) return 1;
    try {
        string mode = argv[1];
        if (mode == "c") compress(argv[2], argv[3]);
        else if (mode == "d") decompress(argv[2], argv[3]);
    } catch (const exception& e) { cerr << e.what() << endl; return 1; }
    return 0;
}
