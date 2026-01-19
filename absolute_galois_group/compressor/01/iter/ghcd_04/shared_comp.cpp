#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <sstream>
#include <Eigen/Dense>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/zlib.hpp>

/**
 * Galois-DCT Hybrid Compressor (GDHC) - v2.0 Optimized
 * Features:
 * 1. D4 Symmetry Group Morphisms (16 virtual atoms per entry)
 * 2. Importance-sampled Dictionary Training
 * 3. Hierarchical Residual DCT
 * * To Compile:
 * g++ -O3 -I /usr/include/eigen3 gdhc_optimized.cpp -o gdhc_optimized -lboost_iostreams -lz
 */

using namespace Eigen;
using namespace std;

// --- Morphism Group Constants ---
// D4 Group: 8 symmetries * 2 polarities = 16 morphisms
enum Morphism : uint8_t {
    ID = 0, ROT90, ROT180, ROT270, FLIP_H, FLIP_V, TRANS, ANTITRANS,
    NEG_ID, NEG_ROT90, NEG_ROT180, NEG_ROT270, NEG_FLIP_H, NEG_FLIP_V, NEG_TRANS, NEG_ANTITRANS
};

// --- Configuration ---
struct CompressionConfig {
    int dict_block_size = 16;
    int dict_entries = 128;   
    int dct_block_size = 8;
    float dct_step = 4.0f;
    int dct_coeffs = 3; 

    void print() const {
        cout << "--- GDHC v2.0 Configuration ---" << endl;
        cout << "L1 Block: " << dict_block_size << "x" << dict_block_size 
             << " | Dict: " << dict_entries << " (Virtual: " << dict_entries * 16 << ")" << endl;
        cout << "L2 DCT: " << dct_block_size << "x" << dct_block_size 
             << " | Step: " << dct_step << " | Coeffs: " << dct_coeffs << "x" << dct_coeffs << endl;
        cout << "--------------------------------" << endl;
    }
};

struct Image {
    int width, height;
    MatrixXf data;
};

#pragma pack(push, 1)
struct DictionaryEntry {
    uint16_t base_id;
    uint8_t morphism; 
};
#pragma pack(pop)

// --- Math & Morphism Helpers ---

MatrixXf apply_morphism(const MatrixXf& block, uint8_t m) {
    MatrixXf res;
    bool negate = (m >= 8);
    uint8_t op = m % 8;

    switch(op) {
        case ROT90:     res = block.transpose().colwise().reverse(); break;
        case ROT180:    res = block.reverse(); break;
        case ROT270:    res = block.transpose().rowwise().reverse(); break;
        case FLIP_H:    res = block.rowwise().reverse(); break;
        case FLIP_V:    res = block.colwise().reverse(); break;
        case TRANS:     res = block.transpose(); break;
        case ANTITRANS: res = block.transpose().reverse(); break;
        default:        res = block; break;
    }
    return negate ? -res : res;
}

MatrixXf get_dct_matrix(int N) {
    MatrixXf T(N, N);
    float pi = acos(-1.0);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i == 0) T(i, j) = 1.0 / sqrt(N);
            else T(i, j) = sqrt(2.0 / N) * cos((2 * j + 1) * i * pi / (2.0 * N));
        }
    }
    return T;
}

// --- PGM IO ---
Image loadPGM(const string& filename) {
    ifstream file(filename, ios::binary);
    if (!file) { cerr << "Error: Cannot open " << filename << endl; exit(1); }
    string magic; file >> magic;
    if (magic != "P5") { cerr << "Error: Only P5 supported." << endl; exit(1); }
    auto skip = [&](ifstream& f) { while (f >> ws && f.peek() == '#') { string d; getline(f, d); } };
    int w, h, maxVal;
    skip(file); file >> w; skip(file); file >> h; skip(file); file >> maxVal;
    file.ignore(1);
    Image img; img.width = w; img.height = h; img.data.resize(h, w);
    vector<unsigned char> buf(w * h);
    file.read((char*)buf.data(), buf.size());
    for (int i = 0; i < h; ++i)
        for (int j = 0; j < w; ++j)
            img.data(i, j) = (float)buf[i * w + j];
    return img;
}

void savePGM(const string& filename, const Image& img) {
    ofstream file(filename, ios::binary);
    file << "P5\n" << img.width << " " << img.height << "\n255\n";
    vector<unsigned char> buf(img.width * img.height);
    for (int i = 0; i < img.height; ++i)
        for (int j = 0; j < img.width; ++j)
            buf[i * img.width + j] = (unsigned char)std::max(0.f, std::min(255.f, img.data(i, j)));
    file.write((char*)buf.data(), buf.size());
}

// --- Improved Dictionary Logic ---
class GaloisDictionary {
public:
    vector<MatrixXf> atoms;

    void train(const MatrixXf& data, int B, int max_entries) {
        struct Cand { float energy; MatrixXf mat; };
        vector<Cand> candidates;

        // Sampling phase: Find blocks with high information density
        for (int i = 0; i + B <= data.rows(); i += B * 2) {
            for (int j = 0; j + B <= data.cols(); j += B * 2) {
                MatrixXf blk = data.block(i, j, B, B);
                MatrixXf centered = blk.array() - blk.mean();
                float energy = centered.norm();
                if (energy > 1e-3) {
                    candidates.push_back({energy, centered / energy});
                }
            }
        }

        sort(candidates.begin(), candidates.end(), [](const Cand& a, const Cand& b) {
            return a.energy > b.energy;
        });

        // Pruning phase: Geometric redundancy check
        for (auto& c : candidates) {
            if (atoms.size() >= max_entries) break;
            bool redundant = false;
            for (const auto& a : atoms) {
                for (uint8_t m = 0; m < 16; ++m) {
                    float dot = (apply_morphism(a, m).array() * c.mat.array()).sum();
                    if (dot > 0.90f) { redundant = true; break; }
                }
                if (redundant) break;
            }
            if (!redundant) atoms.push_back(c.mat);
        }
    }

    DictionaryEntry find_best(const MatrixXf& target, float& out_norm) {
        float best_corr = -2.0f;
        DictionaryEntry best = {0, 0};
        for (size_t i = 0; i < atoms.size(); ++i) {
            for (uint8_t m = 0; m < 16; ++m) {
                float corr = (apply_morphism(atoms[i], m).array() * target.array()).sum();
                if (corr > best_corr) {
                    best_corr = corr;
                    best = {(uint16_t)i, m};
                }
            }
        }
        out_norm = best_corr;
        return best;
    }
};

// --- Compression / Decompression ---

void compress(const string& inputFile, const string& outputFile, const CompressionConfig& config) {
    Image img = loadPGM(inputFile);
    stringstream bitstream;

    // Header
    bitstream.write((char*)&img.width, sizeof(int));
    bitstream.write((char*)&img.height, sizeof(int));
    bitstream.write((char*)&config.dict_block_size, sizeof(int));
    bitstream.write((char*)&config.dict_entries, sizeof(int));
    bitstream.write((char*)&config.dct_block_size, sizeof(int));
    bitstream.write((char*)&config.dct_step, sizeof(float));
    bitstream.write((char*)&config.dct_coeffs, sizeof(int));

    // L1 Training
    GaloisDictionary dict;
    dict.train(img.data, config.dict_block_size, config.dict_entries);
    uint32_t ds = dict.atoms.size();
    bitstream.write((char*)&ds, sizeof(uint32_t));
    for (const auto& a : dict.atoms) {
        bitstream.write((char*)a.data(), a.size() * sizeof(float));
    }

    // L1 Encoding
    MatrixXf recon = MatrixXf::Zero(img.height, img.width);
    int B = config.dict_block_size;
    for (int i = 0; i + B <= img.height; i += B) {
        for (int j = 0; j + B <= img.width; j += B) {
            MatrixXf blk = img.data.block(i, j, B, B);
            float mean = blk.mean();
            MatrixXf centered = blk.array() - mean;
            float norm = centered.norm();
            if (norm > 1e-6) centered /= norm;

            float match_scale;
            DictionaryEntry entry = dict.find_best(centered, match_scale);
            uint8_t q_mean = (uint8_t)std::max(0.f, std::min(255.f, mean));
            uint8_t q_norm = (uint8_t)std::max(0.f, std::min(255.f, norm * 0.5f));

            bitstream.write((char*)&entry, sizeof(DictionaryEntry));
            bitstream.write((char*)&q_mean, 1);
            bitstream.write((char*)&q_norm, 1);

            recon.block(i, j, B, B) = (apply_morphism(dict.atoms[entry.base_id], entry.morphism) * (q_norm * 2.0f)).array() + q_mean;
        }
    }

    // L2 Residual DCT
    MatrixXf residual = img.data - recon;
    MatrixXf T = get_dct_matrix(config.dct_block_size);
    int db = config.dct_block_size;
    for (int i = 0; i + db <= img.height; i += db) {
        for (int j = 0; j + db <= img.width; j += db) {
            MatrixXf rd = residual.block(i, j, db, db);
            MatrixXf coeff = T * rd * T.transpose();
            for (int r = 0; r < config.dct_coeffs; ++r) {
                for (int c = 0; c < config.dct_coeffs; ++c) {
                    int8_t q = (int8_t)std::max(-128, std::min(127, (int)round(coeff(r, c) / config.dct_step)));
                    bitstream.write((char*)&q, 1);
                }
            }
        }
    }

    // Zlib push
    bitstream.seekg(0, ios::beg);
    ofstream outFile(outputFile, ios::binary);
    boost::iostreams::filtering_streambuf<boost::iostreams::output> out;
    out.push(boost::iostreams::zlib_compressor());
    out.push(outFile);
    boost::iostreams::copy(bitstream, out);
    cout << "[*] Compressed to: " << outputFile << endl;
}

void decompress(const string& inputFile, const string& outputFile) {
    ifstream inFile(inputFile, ios::binary);
    stringstream bitstream;
    boost::iostreams::filtering_streambuf<boost::iostreams::input> in;
    in.push(boost::iostreams::zlib_decompressor());
    in.push(inFile);
    boost::iostreams::copy(in, bitstream);

    int w, h;
    CompressionConfig cfg;
    bitstream.read((char*)&w, sizeof(int));
    bitstream.read((char*)&h, sizeof(int));
    bitstream.read((char*)&cfg.dict_block_size, sizeof(int));
    bitstream.read((char*)&cfg.dict_entries, sizeof(int));
    bitstream.read((char*)&cfg.dct_block_size, sizeof(int));
    bitstream.read((char*)&cfg.dct_step, sizeof(float));
    bitstream.read((char*)&cfg.dct_coeffs, sizeof(int));

    uint32_t ds; bitstream.read((char*)&ds, sizeof(uint32_t));
    vector<MatrixXf> atoms(ds, MatrixXf(cfg.dict_block_size, cfg.dict_block_size));
    for (uint32_t i = 0; i < ds; ++i) {
        bitstream.read((char*)atoms[i].data(), atoms[i].size() * sizeof(float));
    }

    Image img; img.width = w; img.height = h; img.data.setZero(h, w);
    int B = cfg.dict_block_size;
    for (int i = 0; i + B <= h; i += B) {
        for (int j = 0; j + B <= w; j += B) {
            DictionaryEntry entry; bitstream.read((char*)&entry, sizeof(DictionaryEntry));
            uint8_t q_m, q_n; bitstream.read((char*)&q_m, 1); bitstream.read((char*)&q_n, 1);
            img.data.block(i, j, B, B) = (apply_morphism(atoms[entry.base_id], entry.morphism) * (q_n * 2.0f)).array() + q_m;
        }
    }

    MatrixXf T = get_dct_matrix(cfg.dct_block_size);
    int db = cfg.dct_block_size;
    for (int i = 0; i + db <= h; i += db) {
        for (int j = 0; j + db <= w; j += db) {
            MatrixXf coeff = MatrixXf::Zero(db, db);
            for (int r = 0; r < cfg.dct_coeffs; ++r) {
                for (int c = 0; c < cfg.dct_coeffs; ++c) {
                    int8_t q; bitstream.read((char*)&q, 1);
                    coeff(r, c) = q * cfg.dct_step;
                }
            }
            img.data.block(i, j, db, db) += T.transpose() * coeff * T;
        }
    }
    savePGM(outputFile, img);
    cout << "[*] Decompressed to: " << outputFile << endl;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        cerr << "Usage: " << argv[0] << " <c/d> <in> <out> [dict_blk dict_size dct_blk dct_step dct_coeffs]" << endl;
        return 1;
    }
    string mode = argv[1];
    if (mode == "c") {
        CompressionConfig cfg;
        if (argc >= 5) cfg.dict_block_size = stoi(argv[4]);
        if (argc >= 6) cfg.dict_entries = stoi(argv[5]);
        if (argc >= 7) cfg.dct_block_size = stoi(argv[6]);
        if (argc >= 8) cfg.dct_step = stof(argv[7]);
        if (argc >= 9) cfg.dct_coeffs = stoi(argv[8]);
        cfg.print();
        compress(argv[2], argv[3], cfg);
    } else {
        decompress(argv[2], argv[3]);
    }
    return 0;
}
