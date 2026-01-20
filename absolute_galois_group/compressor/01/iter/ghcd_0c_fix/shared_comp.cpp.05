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
 * Galois-DCT Hybrid Compressor (GDHC) - v3.0 "Algebraic Extension"
 * * Philosophy:
 * - Ground Field: Dictionary Atoms (Invariants)
 * - Galois Group: Morphisms (Extensions/Automorphisms)
 * - Cohomology: Scale/Offset adjustments (Twists to align ideal structures)
 */

using namespace Eigen;
using namespace std;

// --- Algebraic Morphism Group ---
// Expanded to 32 automorphisms including phase shifts and non-linear mappings
enum Morphism : uint8_t {
    // Standard D4 Symmetries (0-7)
    ID = 0, ROT90, ROT180, ROT270, FLIP_H, FLIP_V, TRANS, ANTITRANS,
    // Frequency Inversions (8-15) - High/Low pass "twists"
    INV_H = 8, INV_V, INV_DIAG, GRAD_X, GRAD_Y, QUAD_1, QUAD_2, QUAD_3,
    // Negative Polarity Extensions (16-31)
    NEG_OFFSET = 16 
};

// --- Math & Algebraic Helpers ---

MatrixXf apply_morphism(const MatrixXf& block, uint8_t m) {
    MatrixXf res;
    bool negate = (m >= NEG_OFFSET);
    uint8_t op = m % NEG_OFFSET;
    int N = block.rows();

    // 1. Geometric Base Automorphisms
    if (op < 8) {
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
    } 
    // 2. Extension Automorphisms (Field-specific twists)
    else {
        res = block;
        switch(op) {
            case INV_H:    res.rowwise().reverseInPlace(); break;
            case INV_V:    res.colwise().reverseInPlace(); break;
            case GRAD_X:   { 
                VectorXf ramp = VectorXf::LinSpaced(N, 0.5f, 1.5f);
                res = res.array().rowwise() * ramp.transpose().array(); 
            } break;
            case GRAD_Y:   {
                VectorXf ramp = VectorXf::LinSpaced(N, 0.5f, 1.5f);
                res = res.array().colwise() * ramp.array();
            } break;
            case QUAD_1:   res = res.array().pow(1.2f); break; // Non-linear extension
            case QUAD_2:   res = res.array().sqrt(); break;    // Curvature extension
            default: break;
        }
    }
    
    if (negate) return -res;
    return res;
}

struct CompressionConfig {
    int dict_block_size = 16;
    int dict_entries = 128;   
    int dct_block_size = 8;
    float dct_step = 4.0f;
    int dct_coeffs = 4; 

    void print() const {
        cout << "--- GDHC v3.0 Algebraic Edition ---" << endl;
        cout << "Ground Field (L1): " << dict_block_size << "x" << dict_block_size 
             << " | Basis: " << dict_entries << " atoms" << endl;
        cout << "Galois Group: 32 automorphisms per atom" << endl;
        cout << "DCT Residual (L2): " << dct_block_size << "x" << dct_block_size << endl;
        cout << "------------------------------------" << endl;
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

// --- Algebraic Dictionary Logic ---
class GaloisDictionary {
public:
    vector<MatrixXf> atoms;

    void train(const MatrixXf& data, int B, int max_entries) {
        struct Cand { float variance; MatrixXf mat; };
        vector<Cand> candidates;

        for (int i = 0; i + B <= data.rows(); i += B) {
            for (int j = 0; j + B <= data.cols(); j += B) {
                MatrixXf blk = data.block(i, j, B, B);
                MatrixXf centered = blk.array() - blk.mean();
                float var = centered.norm();
                if (var > 0.5f) {
                    candidates.push_back({var, centered / var});
                }
            }
        }

        // Sort by significance (the "Ideal" basis)
        sort(candidates.begin(), candidates.end(), [](const Cand& a, const Cand& b) {
            return a.variance > b.variance;
        });

        for (auto& c : candidates) {
            if (atoms.size() >= max_entries) break;
            bool redundant = false;
            for (const auto& a : atoms) {
                // Check if candidate is in the extension of existing atoms
                for (uint8_t m = 0; m < 32; ++m) {
                    float dot = (apply_morphism(a, m).array() * c.mat.array()).sum();
                    if (std::abs(dot) > 0.85f) { redundant = true; break; }
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
            for (uint8_t m = 0; m < 32; ++m) {
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

// --- Compression Logic ---

void compress(const string& inputFile, const string& outputFile, const CompressionConfig& config) {
    Image img = loadPGM(inputFile);
    stringstream bitstream;

    // Header Serialization
    bitstream.write((char*)&img.width, 4);
    bitstream.write((char*)&img.height, 4);
    bitstream.write((char*)&config.dict_block_size, 4);
    bitstream.write((char*)&config.dict_entries, 4);
    bitstream.write((char*)&config.dct_block_size, 4);
    bitstream.write((char*)&config.dct_step, 4);
    bitstream.write((char*)&config.dct_coeffs, 4);

    // Dictionary (Ground Field)
    GaloisDictionary dict;
    dict.train(img.data, config.dict_block_size, config.dict_entries);
    uint32_t ds = dict.atoms.size();
    bitstream.write((char*)&ds, 4);
    for (const auto& a : dict.atoms) bitstream.write((char*)a.data(), a.size() * sizeof(float));

    MatrixXf recon = MatrixXf::Zero(img.height, img.width);
    int B = config.dict_block_size;
    
    // Pass 1: Solve for Cohomology (Scale/Offset/Morphism)
    for (int i = 0; i + B <= img.height; i += B) {
        for (int j = 0; j + B <= img.width; j += B) {
            MatrixXf blk = img.data.block(i, j, B, B);
            float mean = blk.mean();
            MatrixXf centered = blk.array() - mean;
            float norm = centered.norm();
            if (norm > 1e-6) centered /= norm;

            float match_scale;
            DictionaryEntry entry = dict.find_best(centered, match_scale);
            
            // Quantize Cohomology classes
            uint8_t q_mean = (uint8_t)std::max(0.f, std::min(255.f, mean));
            uint8_t q_norm = (uint8_t)std::max(0.f, std::min(255.f, norm * 0.5f));

            bitstream.write((char*)&entry, sizeof(DictionaryEntry));
            bitstream.write((char*)&q_mean, 1);
            bitstream.write((char*)&q_norm, 1);

            // Reconstruct block using the specific field extension
            recon.block(i, j, B, B) = (apply_morphism(dict.atoms[entry.base_id], entry.morphism) * (q_norm * 2.0f)).array() + q_mean;
        }
    }

    // Pass 2: High-Frequency Residual Correction (DCT)
    MatrixXf residual = img.data - recon;
    MatrixXf T = (MatrixXf(config.dct_block_size, config.dct_block_size));
    float pi = acos(-1.0);
    for(int r=0; r<config.dct_block_size; ++r)
        for(int c=0; c<config.dct_block_size; ++c)
            T(r,c) = (r==0)? 1.0/sqrt(config.dct_block_size) : sqrt(2.0/config.dct_block_size)*cos((2*c+1)*r*pi/(2.0*config.dct_block_size));

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

    // Zlib Pipeline
    bitstream.seekg(0, ios::beg);
    ofstream outFile(outputFile, ios::binary);
    boost::iostreams::filtering_streambuf<boost::iostreams::output> out;
    out.push(boost::iostreams::zlib_compressor());
    out.push(outFile);
    boost::iostreams::copy(bitstream, out);
    cout << "[*] Compression Complete. " << endl;
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
    bitstream.read((char*)&w, 4); bitstream.read((char*)&h, 4);
    bitstream.read((char*)&cfg.dict_block_size, 4);
    bitstream.read((char*)&cfg.dict_entries, 4);
    bitstream.read((char*)&cfg.dct_block_size, 4);
    bitstream.read((char*)&cfg.dct_step, 4);
    bitstream.read((char*)&cfg.dct_coeffs, 4);

    uint32_t ds; bitstream.read((char*)&ds, 4);
    vector<MatrixXf> atoms(ds, MatrixXf(cfg.dict_block_size, cfg.dict_block_size));
    for (uint32_t i = 0; i < ds; ++i) bitstream.read((char*)atoms[i].data(), atoms[i].size() * sizeof(float));

    Image img; img.width = w; img.height = h; img.data.setZero(h, w);
    int B = cfg.dict_block_size;
    for (int i = 0; i + B <= h; i += B) {
        for (int j = 0; j + B <= w; j += B) {
            DictionaryEntry entry; bitstream.read((char*)&entry, sizeof(DictionaryEntry));
            uint8_t q_m, q_n; bitstream.read((char*)&q_m, 1); bitstream.read((char*)&q_n, 1);
            img.data.block(i, j, B, B) = (apply_morphism(atoms[entry.base_id], entry.morphism) * (q_n * 2.0f)).array() + q_m;
        }
    }

    // DCT Inverse
    float pi = acos(-1.0);
    MatrixXf T(cfg.dct_block_size, cfg.dct_block_size);
    for(int r=0; r<cfg.dct_block_size; ++r)
        for(int c=0; c<cfg.dct_block_size; ++c)
            T(r,c) = (r==0)? 1.0/sqrt(cfg.dct_block_size) : sqrt(2.0/cfg.dct_block_size)*cos((2*c+1)*r*pi/(2.0*cfg.dct_block_size));

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
    cout << "[*] Decompression Complete." << endl;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        cerr << "GDHC v3.0 Algebraic - Usage: " << argv[0] << " <c/d> <in> <out> [args...]" << endl;
        return 1;
    }
    if (string(argv[1]) == "c") {
        CompressionConfig cfg;
        if (argc >= 5) cfg.dict_block_size = stoi(argv[4]);
        if (argc >= 6) cfg.dict_entries = stoi(argv[5]);
        if (argc >= 7) cfg.dct_block_size = stoi(argv[6]);
        if (argc >= 8) cfg.dct_step = stof(argv[7]);
        if (argc >= 9) cfg.dct_coeffs = stoi(argv[8]);
        cfg.print();
        compress(argv[2], argv[3], cfg);
    } else decompress(argv[2], argv[3]);
    return 0;
}
