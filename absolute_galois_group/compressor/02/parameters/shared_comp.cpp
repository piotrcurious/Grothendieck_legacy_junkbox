#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <string>
#include <map>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/zlib.hpp>

/**
 * Topological Algebraic Compressor (TAC) - Tunable Version
 * Compile:
 * g++ -O3 -I /usr/include/eigen3 tac_compressor.cpp -o tac -lboost_iostreams -lz -fopenmp
 */

using namespace Eigen;
using namespace std;

// --- Configurable Parameters ---
struct Config {
    int canonical_size = 8;
    int dict_size = 256;
    int min_block = 4;
    int max_block = 32;
    float var_threshold = 15.0f;
    int kmeans_iters = 10;
    
    // Derived values
    int get_canonical_dim() const { return canonical_size * canonical_size; }
};

struct Image {
    int width, height;
    MatrixXf data;
};

struct CompressedData {
    int width, height;
    Config cfg; // Store config used for compression
    MatrixXf dictionary;
    std::vector<bool> split_tree;
    std::vector<uint16_t> indices; // FIXED: Changed from uint8_t to uint16_t to support dict_size > 256
    std::vector<uint8_t> scales;
    std::vector<uint8_t> offsets;
};

// --- Helper: Resampling ---
VectorXf canonicalize(const MatrixXf& block, const Config& cfg) {
    int dim = cfg.get_canonical_dim();
    int sz = cfg.canonical_size;

    if (block.rows() == sz && block.cols() == sz) {
        return Map<const VectorXf>(block.data(), dim);
    }
    
    MatrixXf resized(sz, sz);
    float r_step = (float)block.rows() / sz;
    float c_step = (float)block.cols() / sz;

    for (int i = 0; i < sz; ++i) {
        for (int j = 0; j < sz; ++j) {
            int r = std::min((int)(i * r_step), (int)block.rows() - 1);
            int c = std::min((int)(j * c_step), (int)block.cols() - 1);
            resized(i, j) = block(r, c);
        }
    }
    return Map<VectorXf>(resized.data(), dim);
}

MatrixXf reconstruction_map(const VectorXf& vec, int target_rows, int target_cols, const Config& cfg) {
    int sz = cfg.canonical_size;
    MatrixXf canonical = Map<const MatrixXf>(vec.data(), sz, sz);
    
    if (target_rows == sz && target_cols == sz) return canonical;

    MatrixXf out(target_rows, target_cols);
    float r_ratio = (float)(sz - 1) / (std::max(1, target_rows - 1));
    float c_ratio = (float)(sz - 1) / (std::max(1, target_cols - 1));

    for (int i = 0; i < target_rows; ++i) {
        for (int j = 0; j < target_cols; ++j) {
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
    if (!file.is_open()) throw runtime_error("Could not open input file: " + filename);

    string magic; file >> magic;
    if (magic != "P5") throw runtime_error("Invalid PGM format (expect P5)");

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
MatrixXf learn_dictionary(const Image& img, const Config& cfg) {
    cout << "[*] Learning Dictionary (Size=" << cfg.dict_size << ", Manifold=" << cfg.canonical_size << ")..." << endl;
    vector<VectorXf, aligned_allocator<VectorXf>> samples;
    
    int steps[] = {cfg.min_block, cfg.canonical_size, cfg.max_block};
    for (int s : steps) {
        for (int i = 0; i + s <= img.height; i += s) {
            for (int j = 0; j + s <= img.width; j += s) {
                if (rand() % 4 != 0) continue; 
                MatrixXf blk = img.data.block(i, j, s, s);
                VectorXf vec = canonicalize(blk, cfg);
                float mean = vec.mean();
                vec.array() -= mean;
                float norm = vec.norm();
                if (norm > 1e-4) vec /= norm;
                samples.push_back(vec);
            }
        }
    }

    if (samples.empty()) throw runtime_error("Sample set empty. Image too small?");

    MatrixXf dictionary(cfg.get_canonical_dim(), cfg.dict_size);
    mt19937 rng(42);
    uniform_int_distribution<int> dist(0, samples.size() - 1);
    for (int k = 0; k < cfg.dict_size; ++k) dictionary.col(k) = samples[dist(rng)];

    for (int iter = 0; iter < cfg.kmeans_iters; ++iter) {
        vector<VectorXf, aligned_allocator<VectorXf>> centroids(cfg.dict_size, VectorXf::Zero(cfg.get_canonical_dim()));
        vector<int> counts(cfg.dict_size, 0);

        #pragma omp parallel 
        {
            vector<VectorXf, aligned_allocator<VectorXf>> local_centroids(cfg.dict_size, VectorXf::Zero(cfg.get_canonical_dim()));
            vector<int> local_counts(cfg.dict_size, 0);

            #pragma omp for nowait
            for (size_t v = 0; v < samples.size(); ++v) {
                int best_k;
                (dictionary.transpose() * samples[v]).maxCoeff(&best_k);
                local_centroids[best_k] += samples[v];
                local_counts[best_k]++;
            }

            #pragma omp critical
            for (int k = 0; k < cfg.dict_size; ++k) {
                centroids[k] += local_centroids[k];
                counts[k] += local_counts[k];
            }
        }

        float shift = 0;
        for (int k = 0; k < cfg.dict_size; ++k) {
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
    const Config& cfg = storage.cfg;
    
    float mean = img_block.mean();
    float variance = (img_block.array() - mean).square().sum() / (w * h);

    bool should_split = (variance > cfg.var_threshold && (h > cfg.min_block && w > cfg.min_block)) 
                        || (h > cfg.max_block || w > cfg.max_block);

    storage.split_tree.push_back(should_split);

    if (should_split) {
        int half_h = h / 2;
        int half_w = w / 2;
        process_quadrant(img_block.block(0, 0, half_h, half_w), dict, storage);
        process_quadrant(img_block.block(0, half_w, half_h, w - half_w), dict, storage);
        process_quadrant(img_block.block(half_h, 0, h - half_h, half_w), dict, storage);
        process_quadrant(img_block.block(half_h, half_w, h - half_h, w - half_w), dict, storage);
    } else {
        VectorXf vec = canonicalize(img_block, cfg);
        float beta = vec.mean();
        vec.array() -= beta;
        float alpha = vec.norm();
        if (alpha > 1e-5) vec /= alpha;

        int best_k;
        (dict.transpose() * vec).maxCoeff(&best_k);

        // FIXED: best_k can now safely exceed 255
        storage.indices.push_back((uint16_t)best_k);
        storage.scales.push_back((uint8_t)std::clamp(alpha * 0.5f, 0.f, 255.f));
        storage.offsets.push_back((uint8_t)std::clamp(beta, 0.f, 255.f));
    }
}

void compress(const string& inputFile, const string& outputFile, const Config& cfg) {
    Image img = loadPGM(inputFile);
    
    // Align to min_block
    if (img.width % cfg.min_block != 0 || img.height % cfg.min_block != 0) {
        img.width = (img.width / cfg.min_block) * cfg.min_block;
        img.height = (img.height / cfg.min_block) * cfg.min_block;
        img.data = img.data.block(0,0,img.height, img.width).eval();
    }

    MatrixXf dictionary = learn_dictionary(img, cfg);
    CompressedData data;
    data.width = img.width;
    data.height = img.height;
    data.cfg = cfg;
    data.dictionary = dictionary;

    cout << "[*] Encoding Quadtree (Thresh=" << cfg.var_threshold << ")..." << endl;
    process_quadrant(img.data, dictionary, data);

    ofstream outFile(outputFile, ios::binary);
    boost::iostreams::filtering_streambuf<boost::iostreams::output> out;
    out.push(boost::iostreams::zlib_compressor());
    out.push(outFile);
    ostream os(&out);

    // Header Serialization
    os.write((char*)&data.width, sizeof(int));
    os.write((char*)&data.height, sizeof(int));
    os.write((char*)&data.cfg.canonical_size, sizeof(int));
    os.write((char*)&data.cfg.dict_size, sizeof(int));
    
    os.write((char*)data.dictionary.data(), data.dictionary.size() * sizeof(float));

    int tree_size = data.split_tree.size();
    os.write((char*)&tree_size, sizeof(int));
    
    vector<uint8_t> packed_tree((tree_size + 7) / 8, 0);
    for(int i=0; i<tree_size; ++i) {
        if(data.split_tree[i]) packed_tree[i/8] |= (1 << (i%8));
    }
    os.write((char*)packed_tree.data(), packed_tree.size());

    // FIXED: Write uint16_t array (indices)
    os.write((char*)data.indices.data(), data.indices.size() * sizeof(uint16_t));
    os.write((char*)data.scales.data(), data.scales.size());
    os.write((char*)data.offsets.data(), data.offsets.size());

    cout << "[*] Compressed. Nodes: " << tree_size << " Leaves: " << data.indices.size() << endl;
}

// --- Decompression ---
struct StreamReader {
    vector<bool> split_tree;
    vector<uint16_t> indices; // FIXED: Changed from uint8_t to uint16_t
    vector<uint8_t> scales;
    vector<uint8_t> offsets;
    int tree_idx = 0, leaf_idx = 0;
};

void reconstruct_quadrant(Eigen::Ref<MatrixXf> img_block, const MatrixXf& dict, StreamReader& sr, const Config& cfg) {
    if (sr.tree_idx >= (int)sr.split_tree.size()) return;
    
    bool is_split = sr.split_tree[sr.tree_idx++];
    int h = img_block.rows();
    int w = img_block.cols();

    if (is_split) {
        int half_h = h / 2;
        int half_w = w / 2;
        reconstruct_quadrant(img_block.block(0, 0, half_h, half_w), dict, sr, cfg);
        reconstruct_quadrant(img_block.block(0, half_w, half_h, w - half_w), dict, sr, cfg);
        reconstruct_quadrant(img_block.block(half_h, 0, h - half_h, half_w), dict, sr, cfg);
        reconstruct_quadrant(img_block.block(half_h, half_w, h - half_h, w - half_w), dict, sr, cfg);
    } else {
        if (sr.leaf_idx >= (int)sr.indices.size()) return;
        
        uint16_t idx = sr.indices[sr.leaf_idx];
        uint8_t sc = sr.scales[sr.leaf_idx];
        uint8_t off = sr.offsets[sr.leaf_idx];
        sr.leaf_idx++;

        float alpha = (float)sc * 2.0f;
        float beta = (float)off;

        VectorXf vec = dict.col(idx);
        vec = (vec * alpha).array() + beta;
        img_block = reconstruction_map(vec, h, w, cfg);
    }
}

void decompress(const string& inputFile, const string& outputFile) {
    ifstream inFile(inputFile, ios::binary);
    if (!inFile) throw runtime_error("Cannot open input file");

    boost::iostreams::filtering_streambuf<boost::iostreams::input> in;
    in.push(boost::iostreams::zlib_decompressor());
    in.push(inFile);
    istream is(&in);

    int w, h;
    Config cfg;
    is.read((char*)&w, sizeof(int));
    is.read((char*)&h, sizeof(int));
    is.read((char*)&cfg.canonical_size, sizeof(int));
    is.read((char*)&cfg.dict_size, sizeof(int));

    MatrixXf dict(cfg.get_canonical_dim(), cfg.dict_size);
    is.read((char*)dict.data(), dict.size() * sizeof(float));

    int tree_size;
    is.read((char*)&tree_size, sizeof(int));
    vector<uint8_t> packed_tree((tree_size + 7) / 8);
    is.read((char*)packed_tree.data(), packed_tree.size());

    StreamReader sr;
    sr.split_tree.resize(tree_size);
    int leaves = 0;
    for(int i=0; i<tree_size; ++i) {
        sr.split_tree[i] = (packed_tree[i/8] >> (i%8)) & 1;
        if (!sr.split_tree[i]) leaves++;
    }

    sr.indices.resize(leaves);
    sr.scales.resize(leaves);
    sr.offsets.resize(leaves);
    // FIXED: Read uint16_t array
    is.read((char*)sr.indices.data(), leaves * sizeof(uint16_t));
    is.read((char*)sr.scales.data(), leaves);
    is.read((char*)sr.offsets.data(), leaves);

    Image img; img.width = w; img.height = h;
    img.data.resize(h, w);
    reconstruct_quadrant(img.data, dict, sr, cfg);

    savePGM(outputFile, img);
    cout << "[*] Decompression complete." << endl;
}

void print_help(char* name) {
    cout << "Topological Algebraic Compressor (TAC)\n";
    cout << "Usage: " << name << " <mode: c/d> <input> <output> [options]\n";
    cout << "Options (key=value):\n";
    cout << "  --size      : Canonical manifold size (default: 8)\n";
    cout << "  --atoms     : Dictionary atom count (default: 256)\n";
    cout << "  --thresh    : Variance threshold for splitting (default: 15.0)\n";
    cout << "  --min       : Minimum block size (default: 4)\n";
    cout << "  --max       : Maximum block size (default: 32)\n";
    cout << "  --iters     : K-Means training iterations (default: 10)\n";
    cout << "Example: " << name << " c in.pgm out.tac --thresh=5.5 --atoms=1024\n";
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        print_help(argv[0]);
        return 1;
    }

    string mode = argv[1];
    string input = argv[2];
    string output = argv[3];
    Config cfg;

    for (int i = 4; i < argc; ++i) {
        string arg = argv[i];
        size_t eq = arg.find('=');
        if (eq == string::npos) continue;
        string key = arg.substr(0, eq);
        string val = arg.substr(eq + 1);

        if (key == "--size") cfg.canonical_size = stoi(val);
        else if (key == "--atoms") cfg.dict_size = stoi(val);
        else if (key == "--thresh") cfg.var_threshold = stof(val);
        else if (key == "--min") cfg.min_block = stoi(val);
        else if (key == "--max") cfg.max_block = stoi(val);
        else if (key == "--iters") cfg.kmeans_iters = stoi(val);
    }

    try {
        if (mode == "c") compress(input, output, cfg);
        else if (mode == "d") decompress(input, output);
        else print_help(argv[0]);
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }

    return 0;
}
