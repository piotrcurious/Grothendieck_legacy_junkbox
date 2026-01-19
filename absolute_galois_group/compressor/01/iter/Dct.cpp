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
 * Advanced Algebraic Compressor: Dictionary Learning & Structure Sharing
 * * To Compile:
 * g++ -O3 -I /usr/include/eigen3 shared_galois.cpp -o shared_comp -lboost_iostreams -lz -fopenmp
 */

using namespace Eigen;
using namespace std;

// --- Tuning Parameters ---
const int BLOCK_SIZE = 8;
const int VECTOR_DIM = BLOCK_SIZE * BLOCK_SIZE;
const int DICT_SIZE = 256; 

struct Image {
    int width, height;
    MatrixXf data;
};

struct BlockMap {
    uint8_t basis_index; 
    uint8_t scale;       
    uint8_t offset;      
};

// --- Robust PGM IO ---
Image loadPGM(const string& filename) {
    ifstream file(filename, ios::binary);
    if (!file.is_open()) throw runtime_error("Could not open input file: " + filename);

    string magic;
    file >> magic;
    if (magic != "P5") throw runtime_error("Invalid PGM format (only P5 supported)");

    // Skip comments and whitespace
    auto skipComments = [&file]() {
        char ch;
        while (file >> ws && file.peek() == '#') {
            file.ignore(4096, '\n');
        }
    };

    skipComments();
    int w, h, maxVal;
    if (!(file >> w >> h)) throw runtime_error("Invalid PGM dimensions");
    skipComments();
    if (!(file >> maxVal)) throw runtime_error("Invalid PGM max value");
    
    file.ignore(1); // Skip the single whitespace character after header

    Image img;
    img.width = w;
    img.height = h;
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
    if (!file.is_open()) throw runtime_error("Could not open file for writing: " + filename);

    file << "P5\n" << img.width << " " << img.height << "\n255\n";
    vector<unsigned char> buffer(img.width * img.height);
    for (int i = 0; i < img.height; ++i) {
        for (int j = 0; j < img.width; ++j) {
            float val = img.data(i, j);
            buffer[i * img.width + j] = (unsigned char)std::max(0.f, std::min(255.f, val));
        }
    }
    file.write(reinterpret_cast<char*>(buffer.data()), buffer.size());
}

// --- Dictionary Learning ---
MatrixXf learn_dictionary(const Image& img) {
    cout << "[*] Extracting blocks and normalizing..." << endl;
    
    // Use aligned_allocator to prevent SIGSEGV on some architectures
    vector<VectorXf, aligned_allocator<VectorXf>> vectors;
    
    for (int i = 0; i + BLOCK_SIZE <= img.height; i += BLOCK_SIZE) {
        for (int j = 0; j + BLOCK_SIZE <= img.width; j += BLOCK_SIZE) {
            MatrixXf blk = img.data.block(i, j, BLOCK_SIZE, BLOCK_SIZE);
            VectorXf vec = Map<VectorXf>(blk.data(), VECTOR_DIM);
            
            float mean = vec.mean();
            vec.array() -= mean;
            float norm = vec.norm();
            if (norm > 1e-5) vec /= norm;
            
            vectors.push_back(vec);
        }
    }

    if (vectors.empty()) throw runtime_error("Image too small for block size");

    cout << "[*] Training on " << vectors.size() << " blocks (K-Means)..." << endl;
    MatrixXf dictionary(VECTOR_DIM, DICT_SIZE);
    mt19937 rng(12345);
    uniform_int_distribution<int> dist(0, vectors.size() - 1);
    for (int k = 0; k < DICT_SIZE; ++k) dictionary.col(k) = vectors[dist(rng)];

    for (int iter = 0; iter < 12; ++iter) {
        vector<VectorXf, aligned_allocator<VectorXf>> centroids(DICT_SIZE, VectorXf::Zero(VECTOR_DIM));
        vector<int> counts(DICT_SIZE, 0);

        // Assignment step (Parallelized)
        #pragma omp parallel
        {
            vector<VectorXf, aligned_allocator<VectorXf>> local_centroids(DICT_SIZE, VectorXf::Zero(VECTOR_DIM));
            vector<int> local_counts(DICT_SIZE, 0);

            #pragma omp for nowait
            for (size_t v = 0; v < vectors.size(); ++v) {
                int best_k;
                (dictionary.transpose() * vectors[v]).maxCoeff(&best_k);
                local_centroids[best_k] += vectors[v];
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

        // Update step
        float total_shift = 0;
        for (int k = 0; k < DICT_SIZE; ++k) {
            if (counts[k] > 0) {
                centroids[k] /= (float)counts[k];
                centroids[k].normalize();
                total_shift += (centroids[k] - dictionary.col(k)).norm();
                dictionary.col(k) = centroids[k];
            } else {
                dictionary.col(k) = vectors[dist(rng)];
            }
        }
        cout << "    Iteration " << iter + 1 << " - Shift: " << total_shift << endl;
        if (total_shift < 1e-2) break;
    }
    return dictionary;
}

void compress(const string& inputFile, const string& outputFile) {
    Image img = loadPGM(inputFile);
    MatrixXf dictionary = learn_dictionary(img);

    stringstream bitstream;
    bitstream.write((char*)&img.width, sizeof(int));
    bitstream.write((char*)&img.height, sizeof(int));
    bitstream.write((char*)dictionary.data(), dictionary.size() * sizeof(float));

    cout << "[*] Mapping Blocks..." << endl;
    for (int i = 0; i + BLOCK_SIZE <= img.height; i += BLOCK_SIZE) {
        for (int j = 0; j + BLOCK_SIZE <= img.width; j += BLOCK_SIZE) {
            MatrixXf blk = img.data.block(i, j, BLOCK_SIZE, BLOCK_SIZE);
            VectorXf vec = Map<VectorXf>(blk.data(), VECTOR_DIM);

            float beta = vec.mean();
            vec.array() -= beta;
            float alpha = vec.norm();
            if (alpha > 1e-5) vec /= alpha;

            int best_k;
            (dictionary.transpose() * vec).maxCoeff(&best_k);
            
            uint8_t q_scale = (uint8_t)std::min(255.f, std::max(0.f, alpha * 0.5f)); 
            uint8_t q_offset = (uint8_t)std::min(255.f, std::max(0.f, beta));

            BlockMap bmap = {(uint8_t)best_k, q_scale, q_offset};
            bitstream.write((char*)&bmap, sizeof(BlockMap));
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

    MatrixXf dictionary(VECTOR_DIM, DICT_SIZE);
    bitstream.read((char*)dictionary.data(), dictionary.size() * sizeof(float));

    Image img; img.width = w; img.height = h; img.data.setZero(h, w);

    cout << "[*] Reconstructing..." << endl;
    for (int i = 0; i + BLOCK_SIZE <= h; i += BLOCK_SIZE) {
        for (int j = 0; j + BLOCK_SIZE <= w; j += BLOCK_SIZE) {
            BlockMap bmap;
            bitstream.read((char*)&bmap, sizeof(BlockMap));

            float alpha = (float)bmap.scale * 2.0f; 
            float beta = (float)bmap.offset;
            VectorXf vec = dictionary.col(bmap.basis_index);
            vec = (vec * alpha).array() + beta;

            img.data.block(i, j, BLOCK_SIZE, BLOCK_SIZE) = Map<MatrixXf>(vec.data(), BLOCK_SIZE, BLOCK_SIZE);
        }
    }
    savePGM(outputFile, img);
    cout << "[*] Decompression Complete: " << outputFile << endl;
}

void printUsage(const char* name) {
    cout << "Usage: " << name << " <mode> <input> <output>\n"
         << "Modes:\n"
         << "  c : Compress PGM to binary\n"
         << "  d : Decompress binary to PGM\n";
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        printUsage(argv[0]);
        return 1;
    }

    try {
        string mode = argv[1];
        if (mode == "c") compress(argv[2], argv[3]);
        else if (mode == "d") decompress(argv[2], argv[3]);
        else {
            printUsage(argv[0]);
            return 1;
        }
    } catch (const exception& e) {
        cerr << "[ERROR] " << e.what() << endl;
        return 1;
    }
    return 0;
}
