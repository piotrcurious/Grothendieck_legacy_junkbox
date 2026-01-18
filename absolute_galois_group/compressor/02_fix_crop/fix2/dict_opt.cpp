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
#include <omp.h>

using namespace Eigen;
using namespace std;

// --- Tuning Parameters ---
const int CANONICAL_SIZE = 8;
const int CANONICAL_DIM = CANONICAL_SIZE * CANONICAL_SIZE;
const int DICT_SIZE = 256; 
const int MIN_BLOCK = 4;
const int MAX_BLOCK = 64;
const float VAR_THRESHOLD = 150.0f; // Lowered slightly for higher quality

struct Image {
    int width, height;
    MatrixXf data;
};

// --- Automorphism Subsystem (D4 Group) ---
// Maps a canonical vector to one of its 8 isometries
// 0: Identity, 1: Rot90, 2: Rot180, 3: Rot270
// 4: FlipX, 5: FlipX+Rot90, 6: FlipX+Rot180, 7: FlipX+Rot270
VectorXf apply_isometry(const VectorXf& vec, int mode) {
    if (mode == 0) return vec;
    MatrixXf blk = Map<const MatrixXf>(vec.data(), CANONICAL_SIZE, CANONICAL_SIZE);
    MatrixXf out(CANONICAL_SIZE, CANONICAL_SIZE);
    
    for(int r=0; r<CANONICAL_SIZE; ++r) {
        for(int c=0; c<CANONICAL_SIZE; ++c) {
            int nr = r, nc = c;
            // Apply Flip first (if mode >= 4)
            if (mode >= 4) nr = (CANONICAL_SIZE - 1) - nr;
            
            // Apply Rotation
            int rot = mode % 4;
            int tr, tc;
            if (rot == 0) { tr = nr; tc = nc; }
            else if (rot == 1) { tr = nc; tc = (CANONICAL_SIZE - 1) - nr; }
            else if (rot == 2) { tr = (CANONICAL_SIZE - 1) - nr; tc = (CANONICAL_SIZE - 1) - nc; }
            else { tr = (CANONICAL_SIZE - 1) - nc; tc = nr; } // rot == 3
            
            out(tr, tc) = blk(r, c);
        }
    }
    return Map<VectorXf>(out.data(), CANONICAL_DIM);
}

struct CompressedData {
    int orig_width, orig_height;
    int padded_width, padded_height;
    MatrixXf dictionary;
    std::vector<bool> split_tree;
    std::vector<uint8_t> indices;
    std::vector<uint8_t> orientations; // New: Stores the isometry index (0-7)
    std::vector<uint8_t> scales;
    std::vector<uint8_t> offsets;
    std::vector<uint16_t> block_sizes;
};

// --- Helper Functions ---
VectorXf canonicalize(const MatrixXf& block) {
    if (block.rows() == CANONICAL_SIZE && block.cols() == CANONICAL_SIZE) {
        return Map<const VectorXf>(block.data(), CANONICAL_DIM);
    }
    // Bilinear Interpolation
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
            resized(i, j) = (1-dr)*(1-dc)*block(r0,c0) + (1-dr)*dc*block(r0,c1) + 
                            dr*(1-dc)*block(r1,c0) + dr*dc*block(r1,c1);
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
            out(i, j) = (1-dr)*(1-dc)*canonical(r0,c0) + (1-dr)*dc*canonical(r0,c1) + 
                        dr*(1-dc)*canonical(r1,c0) + dr*dc*canonical(r1,c1);
        }
    }
    return out;
}

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

// --- Invariant Dictionary Learning ---
MatrixXf learn_dictionary(const Image& img) {
    cout << "[*] Learning Invariant Dictionary (D4 Symmetry)..." << endl;
    vector<VectorXf, aligned_allocator<VectorXf>> samples;
    
    // Heuristic: Multi-scale sampling with higher density
    int steps[] = {4, 8, 12, 16};
    
    // Cryptographic: Use Mersenne Twister for high-quality randomness
    mt19937 rng(1337); 
    
    for (int s : steps) {
        // Statistical: adaptive stride based on block size
        int stride = std::max(4, s / 2);
        for (int i = 0; i + s <= img.height; i += stride) {
            for (int j = 0; j + s <= img.width; j += stride) {
                // Heuristic: Skip flat regions early (variance check)
                // to focus dictionary on edges/textures
                MatrixXf blk = img.data.block(i, j, s, s);
                float var = (blk.array() - blk.mean()).square().mean();
                if (var < 20.0f) continue; 

                if (rng() % 5 != 0) continue; // Subsampling

                VectorXf vec = canonicalize(blk);
                float mean = vec.mean();
                vec.array() -= mean;
                float norm = vec.norm();
                if (norm > 1e-4) vec /= norm;
                else vec.setZero();
                
                samples.push_back(vec);
            }
        }
    }

    if (samples.empty()) throw runtime_error("Image variance too low for dictionary learning");
    cout << "    Collected " << samples.size() << " samples." << endl;

    MatrixXf dictionary(CANONICAL_DIM, DICT_SIZE);
    uniform_int_distribution<int> dist(0, (int)samples.size() - 1);
    for (int k = 0; k < DICT_SIZE; ++k) dictionary.col(k) = samples[dist(rng)];

    // Invariant K-SVD / K-Means
    for (int iter = 0; iter < 20; ++iter) {
        vector<VectorXf, aligned_allocator<VectorXf>> centroids(DICT_SIZE, VectorXf::Zero(CANONICAL_DIM));
        vector<int> counts(DICT_SIZE, 0);
        float total_error = 0;

        #pragma omp parallel 
        {
            vector<VectorXf, aligned_allocator<VectorXf>> local_centroids(DICT_SIZE, VectorXf::Zero(CANONICAL_DIM));
            vector<int> local_counts(DICT_SIZE, 0);

            #pragma omp for nowait
            for (size_t v = 0; v < samples.size(); ++v) {
                int best_k = -1;
                float best_score = -1e9;
                int best_iso = 0;

                // For each atom, check all 8 symmetries of the sample
                // We actually do dot(atom, iso(sample)) which is equivalent to dot(inv_iso(atom), sample)
                // Optimization: Pre-generate symmetries for atoms or samples. 
                // Given DICT_SIZE is small (256), we brute force the 8 transforms here for simplicity.
                
                for (int k = 0; k < DICT_SIZE; ++k) {
                    float score = samples[v].dot(dictionary.col(k));
                    // Check if aligned is better, checking simply score isn't enough for invariance
                    // Actually, we want max |dot|. But for accumulation we need orientation.
                    // Let's brute force: find k and iso that maximizes dot(dictionary.col(k), iso(sample))
                }

                // Optimization: Instead of transforming sample 256*8 times, 
                // transform sample 8 times, then vector-matrix mult.
                MatrixXf transformed_samples(CANONICAL_DIM, 8);
                for(int iso=0; iso<8; ++iso) transformed_samples.col(iso) = apply_isometry(samples[v], iso);
                
                MatrixXf scores = dictionary.transpose() * transformed_samples; // (256 x 8)
                
                MatrixXf::Index maxRow, maxCol;
                float maxVal = scores.maxCoeff(&maxRow, &maxCol);
                
                // Add the aligned sample to the centroid
                local_centroids[maxRow] += transformed_samples.col(maxCol);
                local_counts[maxRow]++;
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
            } else {
                // Re-initialize dead atoms
                dictionary.col(k) = samples[dist(rng)];
                dictionary.col(k).normalize();
                shift += 1.0f;
            }
        }
        if (shift < 1e-3) break;
    }
    return dictionary;
}

void process_quadrant(const Eigen::Ref<const MatrixXf>& img_block, const MatrixXf& dict, CompressedData& storage) {    
    int h = img_block.rows(); int w = img_block.cols();
    
    // Heuristic: Variance based splitting
    float mean = img_block.mean();
    float variance = (img_block.array() - mean).square().mean();
    
    bool can_split = (h >= 2*MIN_BLOCK && w >= 2*MIN_BLOCK);
    bool must_split = (h > MAX_BLOCK || w > MAX_BLOCK);
    // Refined Heuristic: Force split if variance is extremely high (edge) or block is huge
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
        float beta = vec.mean(); 
        vec.array() -= beta; // Remove DC
        
        // Statistical: Least Squares Fit
        // We want min || alpha * atom + beta - block ||^2
        // Since atom is unit norm and block is zero-meaned:
        // alpha = dot(atom, block)
        
        // Generate 8 isometries of the input block
        MatrixXf transformed_vecs(CANONICAL_DIM, 8);
        for(int i=0; i<8; ++i) transformed_vecs.col(i) = apply_isometry(vec, i);
        
        // Match against dictionary: (256 x 8) scores
        MatrixXf scores = dict.transpose() * transformed_vecs;
        
        MatrixXf::Index best_k, best_iso;
        float max_dot = scores.maxCoeff(&best_k, &best_iso);
        
        float alpha = max_dot; // Optimal Least Squares scale
        
        // Store
        storage.indices.push_back((uint8_t)best_k);
        storage.orientations.push_back((uint8_t)best_iso);
        storage.scales.push_back((uint8_t)std::min(255.f, std::max(0.f, alpha * 2.0f))); // Scale factor heuristic
        storage.offsets.push_back((uint8_t)std::min(255.f, std::max(0.f, beta)));
        storage.block_sizes.push_back((uint16_t)((h << 8) | w));
    }
}

void compress(const string& inputFile, const string& outputFile) {
    Image img = loadPGM(inputFile);
    int orig_h = img.height;
    int orig_w = img.width;

    int padded_h = 1, padded_w = 1;
    while (padded_h < orig_h) padded_h *= 2;
    while (padded_w < orig_w) padded_w *= 2;
    
    MatrixXf processed_data = MatrixXf::Zero(padded_h, padded_w);
    processed_data.block(0, 0, orig_h, orig_w) = img.data;

    MatrixXf dictionary = learn_dictionary(img);
    CompressedData data;
    data.orig_width = orig_w;
    data.orig_height = orig_h;
    data.padded_width = padded_w;
    data.padded_height = padded_h;
    data.dictionary = dictionary;

    process_quadrant(processed_data, dictionary, data);

    ofstream outFile(outputFile, ios::binary);
    boost::iostreams::filtering_streambuf<boost::iostreams::output> out;
    out.push(boost::iostreams::zlib_compressor(boost::iostreams::zlib::best_compression)); // Statistical coding
    out.push(outFile);
    ostream os(&out);

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
    os.write((char*)data.orientations.data(), leaf_count); // Write orientations
    os.write((char*)data.scales.data(), leaf_count);
    os.write((char*)data.offsets.data(), leaf_count);
    os.write((char*)data.block_sizes.data(), leaf_count * sizeof(uint16_t));
    
    cout << "[*] Compressed. Leaves: " << leaf_count << endl;
}

struct StreamReader {
    vector<bool> split_tree;
    vector<uint8_t> indices;
    vector<uint8_t> orientations;
    vector<uint8_t> scales;
    vector<uint8_t> offsets;
    vector<uint16_t> block_sizes;
    int tree_idx = 0, leaf_idx = 0;
};

// --- Invariant Reconstruction ---
void reconstruct_quadrant(Eigen::Ref<MatrixXf> img_block, const MatrixXf& dict, StreamReader& sr) {
    if (sr.tree_idx >= (int)sr.split_tree.size()) return;
    bool is_split = sr.split_tree[sr.tree_idx++];
    
    if (is_split) {
        int h = img_block.rows(); int w = img_block.cols();
        int half_h = h / 2; int half_w = w / 2;
        reconstruct_quadrant(img_block.block(0, 0, half_h, half_w), dict, sr);
        reconstruct_quadrant(img_block.block(0, half_w, half_h, w - half_w), dict, sr);
        reconstruct_quadrant(img_block.block(half_h, 0, h - half_h, half_w), dict, sr);
        reconstruct_quadrant(img_block.block(half_h, half_w, h - half_h, w - half_w), dict, sr);
    } else {
        if (sr.leaf_idx >= (int)sr.indices.size()) return;
        uint8_t idx = sr.indices[sr.leaf_idx];
        uint8_t iso = sr.orientations[sr.leaf_idx]; // Get orientation
        uint8_t sc = sr.scales[sr.leaf_idx];
        uint8_t off = sr.offsets[sr.leaf_idx];
        uint16_t packed_size = sr.block_sizes[sr.leaf_idx++];
        
        int block_h = (packed_size >> 8) & 0xFF;
        int block_w = packed_size & 0xFF;
        
        float alpha = (float)sc / 2.0f; 
        float beta = (float)off;
        
        // 1. Fetch Atom
        VectorXf atom = dict.col(idx);
        // 2. Apply Inverse Isometry (For D4, inverse of rot is rot_neg, etc. 
        //    However, our matching logic optimized dot(atom, iso(block)).
        //    Reconstruction is: block approx iso_inv(atom) * alpha + beta?
        //    Wait: Maximize dot(dict_col, ApplyIso(block)). 
        //    So dict_col ~ ApplyIso(block).
        //    => block ~ ApplyInvIso(dict_col).
        //    D4 inverse property: Inv(Iso(k)) depends on k.
        //    Simple check: If I rotated block 90deg to match atom, I must rotate atom -90deg (270) to match block.
        
        // Inverse Mapping for D4 (Indices 0..7)
        // 0->0 (Id)
        // 1->3 (R90 -> R270)
        // 2->2 (R180 -> R180)
        // 3->1 (R270 -> R90)
        // 4->4 (FlipX -> FlipX)
        // 5->5 (FlipX+R90 inv is FlipX+R90) Check: FX R90 * FX R90 = FX R90 FX R90 = FX FX R270 R90 = I. Yes.
        // Actually, for D4 generated by r,s: s^2=1, r^4=1, srs=r^-1.
        // Our encoding:
        // 0: I, 1: r, 2: r^2, 3: r^3
        // 4: s, 5: sr, 6: sr^2, 7: sr^3
        // Inverses:
        // Inv(0)=0, Inv(1)=3, Inv(2)=2, Inv(3)=1
        // Inv(4)=4, Inv(5)=5, Inv(6)=6, Inv(7)=7 (Reflections are self-inverse)
        
        int inv_iso = iso;
        if (iso == 1) inv_iso = 3;
        if (iso == 3) inv_iso = 1;
        
        VectorXf aligned_atom = apply_isometry(atom, inv_iso);
        
        VectorXf vec = (aligned_atom * alpha).array() + beta;
        img_block = reconstruction_map(vec, block_h, block_w);
    }
}

void decompress(const string& inputFile, const string& outputFile) {
    ifstream inFile(inputFile, ios::binary);
    if (!inFile) throw runtime_error("Cannot open input file");
    
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
    sr.indices.resize(leaf_count); 
    sr.orientations.resize(leaf_count);
    sr.scales.resize(leaf_count);
    sr.offsets.resize(leaf_count); 
    sr.block_sizes.resize(leaf_count);
    
    is.read((char*)sr.indices.data(), leaf_count);
    is.read((char*)sr.orientations.data(), leaf_count);
    is.read((char*)sr.scales.data(), leaf_count);
    is.read((char*)sr.offsets.data(), leaf_count);
    is.read((char*)sr.block_sizes.data(), leaf_count * sizeof(uint16_t));

    MatrixXf full_padded = MatrixXf::Zero(padded_h, padded_w);
    reconstruct_quadrant(full_padded, dict, sr);

    Image img;
    img.width = orig_w;
    img.height = orig_h;
    img.data = full_padded.block(0, 0, orig_h, orig_w);

    savePGM(outputFile, img);
    cout << "[*] Decompression Complete." << endl;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        cerr << "Usage: " << argv[0] << " <c|d> <input.pgm> <output.bin/pgm>" << endl;
        return 1;
    }
    try {
        string mode = argv[1];
        if (mode == "c") compress(argv[2], argv[3]);
        else if (mode == "d") decompress(argv[2], argv[3]);
    } catch (const exception& e) { cerr << "Error: " << e.what() << endl; return 1; }
    return 0;
}
