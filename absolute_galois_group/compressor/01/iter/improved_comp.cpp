#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <Eigen/Dense>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/zlib.hpp>

/**
 * Hierarchical Algebraic Residual Compressor (HARC)
 * 
 * This implementation achieves high compression ratios by combining:
 * 1. Progressive Decomposition: Multi-scale analysis (16x16 and 8x8 blocks).
 * 2. Algebraic Methods: Discrete Cosine Transform (DCT) for sparse representation.
 * 3. Hierarchical Residuals: Encoding the error of the coarse level at a finer scale.
 * 
 * Compared to dictionary learning, this method avoids the overhead of storing
 * basis vectors and leverages the energy compaction properties of the DCT.
 * 
 * To Compile:
 * g++ -O3 -I /usr/include/eigen3 improved_comp.cpp -o improved_comp -lboost_iostreams -lz
 */

using namespace Eigen;
using namespace std;

struct Image {
    int width, height;
    MatrixXf data;
};

// --- Algebraic Basis: Discrete Cosine Transform (DCT) ---
// Generates an NxN DCT transformation matrix.
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

// Applies 2D DCT to a block.
MatrixXf apply_dct(const MatrixXf& block, const MatrixXf& T) {
    return T * block * T.transpose();
}

// Applies 2D Inverse DCT to coefficients.
MatrixXf apply_idct(const MatrixXf& coeff, const MatrixXf& T) {
    return T.transpose() * coeff * T;
}

// --- PGM Image IO ---
Image loadPGM(const string& filename) {
    ifstream file(filename, ios::binary);
    if (!file.is_open()) throw runtime_error("Could not open input file");
    string magic; file >> magic;
    if (magic != "P5") throw runtime_error("Only P5 PGM supported");
    
    auto skipComments = [&file]() {
        while (file >> ws && file.peek() == '#') file.ignore(4096, '\n');
    };
    skipComments();
    int w, h, maxVal;
    if (!(file >> w >> h >> maxVal)) throw runtime_error("Invalid PGM header");
    file.ignore(1);

    Image img; img.width = w; img.height = h; img.data.resize(h, w);
    vector<unsigned char> buffer(w * h);
    file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
    for (int i = 0; i < h; ++i)
        for (int j = 0; j < w; ++j)
            img.data(i, j) = static_cast<float>(buffer[i * w + j]);
    return img;
}

void savePGM(const string& filename, const Image& img) {
    ofstream file(filename, ios::binary);
    file << "P5\n" << img.width << " " << img.height << "\n255\n";
    vector<unsigned char> buffer(img.width * img.height);
    for (int i = 0; i < img.height; ++i)
        for (int j = 0; j < img.width; ++j)
            buffer[i * img.width + j] = (unsigned char)max(0.f, min(255.f, img.data(i, j)));
    file.write(reinterpret_cast<char*>(buffer.data()), buffer.size());
}

// --- Compression Logic ---
void compress(const string& inputFile, const string& outputFile) {
    Image img = loadPGM(inputFile);
    int w = img.width, h = img.height;
    stringstream bitstream;
    bitstream.write((char*)&w, sizeof(int));
    bitstream.write((char*)&h, sizeof(int));

    MatrixXf T16 = get_dct_matrix(16);
    MatrixXf T8 = get_dct_matrix(8);
    MatrixXf recon = MatrixXf::Zero(h, w);

    // Level 1: Coarse Decomposition (16x16 blocks)
    // We keep the top 3x3 low-frequency components.
    for (int i = 0; i + 16 <= h; i += 16) {
        for (int j = 0; j + 16 <= w; j += 16) {
            MatrixXf blk = img.data.block(i, j, 16, 16);
            MatrixXf coeff = apply_dct(blk, T16);
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    int16_t q = (int16_t)round(coeff(r, c) / 8.0);
                    bitstream.write((char*)&q, sizeof(int16_t));
                    coeff(r, c) = q * 8.0;
                }
            }
            MatrixXf sparse_coeff = MatrixXf::Zero(16, 16);
            sparse_coeff.block(0, 0, 3, 3) = coeff.block(0, 0, 3, 3);
            recon.block(i, j, 16, 16) = apply_idct(sparse_coeff, T16);
        }
    }

    // Level 2: Hierarchical Residuals (8x8 blocks)
    // We encode the difference between the original and the coarse reconstruction.
    MatrixXf residual = img.data - recon;
    for (int i = 0; i + 8 <= h; i += 8) {
        for (int j = 0; j + 8 <= w; j += 8) {
            MatrixXf blk = residual.block(i, j, 8, 8);
            MatrixXf coeff = apply_dct(blk, T8);
            for (int r = 0; r < 2; ++r) {
                for (int c = 0; c < 2; ++c) {
                    // Residuals are smaller, so we use 8-bit quantization.
                    int8_t q = (int8_t)max(-128, min(127, (int)round(coeff(r, c) / 4.0)));
                    bitstream.write((char*)&q, sizeof(int8_t));
                }
            }
        }
    }

    ofstream outFile(outputFile, ios::binary);
    boost::iostreams::filtering_streambuf<boost::iostreams::output> out;
    out.push(boost::iostreams::zlib_compressor());
    out.push(outFile);
    boost::iostreams::copy(bitstream, out);
    cout << "[*] Compression Complete: " << outputFile << endl;
}

void decompress(const string& inputFile, const string& outputFile) {
    ifstream inFile(inputFile, ios::binary);
    if (!inFile) throw runtime_error("Cannot open compressed file");
    stringstream bitstream;
    boost::iostreams::filtering_streambuf<boost::iostreams::input> in;
    in.push(boost::iostreams::zlib_decompressor());
    in.push(inFile);
    boost::iostreams::copy(in, bitstream);

    int w, h;
    bitstream.read((char*)&w, sizeof(int));
    bitstream.read((char*)&h, sizeof(int));

    Image img; img.width = w; img.height = h; img.data.setZero(h, w);
    MatrixXf T16 = get_dct_matrix(16);
    MatrixXf T8 = get_dct_matrix(8);

    // Reconstruct Level 1
    for (int i = 0; i + 16 <= h; i += 16) {
        for (int j = 0; j + 16 <= w; j += 16) {
            MatrixXf coeff = MatrixXf::Zero(16, 16);
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    int16_t q; bitstream.read((char*)&q, sizeof(int16_t));
                    coeff(r, c) = q * 8.0;
                }
            }
            img.data.block(i, j, 16, 16) = apply_idct(coeff, T16);
        }
    }

    // Reconstruct Level 2 Residuals
    for (int i = 0; i + 8 <= h; i += 8) {
        for (int j = 0; j + 8 <= w; j += 8) {
            MatrixXf coeff = MatrixXf::Zero(8, 8);
            for (int r = 0; r < 2; ++r) {
                for (int c = 0; c < 2; ++c) {
                    int8_t q; bitstream.read((char*)&q, sizeof(int8_t));
                    coeff(r, c) = q * 4.0;
                }
            }
            img.data.block(i, j, 8, 8) += apply_idct(coeff, T8);
        }
    }
    savePGM(outputFile, img);
    cout << "[*] Decompression Complete: " << outputFile << endl;
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        cout << "Usage: " << argv[0] << " <mode> <input> <output>\n"
             << "Modes: c (compress), d (decompress)\n";
        return 1;
    }
    try {
        if (string(argv[1]) == "c") compress(argv[2], argv[3]);
        else if (string(argv[1]) == "d") decompress(argv[2], argv[3]);
        else return 1;
    } catch (const exception& e) {
        cerr << "[ERROR] " << e.what() << endl;
        return 1;
    }
    return 0;
}
