#include "rnn_absolute_core.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <string>

struct Image {
    int width, height;
    std::vector<uint8_t> data;
};

Image loadPGM(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Could not open input file: " + filename);
    std::string magic;
    file >> magic;
    if (magic != "P5") throw std::runtime_error("Only P5 PGM supported");
    int w, h, maxVal;
    file >> w >> h >> maxVal;
    file.ignore(1);
    Image img;
    img.width = w; img.height = h;
    img.data.resize(w * h);
    file.read(reinterpret_cast<char*>(img.data.data()), img.data.size());
    return img;
}

int main(int argc, char** argv) {
    std::string input_pgm = "absolute_galois_group/compressor/01/test.pgm";
    if (argc > 1) input_pgm = argv[1];

    std::cout << "--- RNN Absolute Galois Group Compressor [PGM Test] ---" << std::endl;

    Image img;
    try {
        img = loadPGM(input_pgm);
    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    const int block_size = 8;
    std::vector<std::vector<double>> sequences;
    for (size_t i = 0; i + block_size <= img.data.size(); i += block_size) {
        std::vector<double> seq;
        for (int j = 0; j < block_size; ++j) seq.push_back(static_cast<double>(img.data[i+j]));
        sequences.push_back(seq);
    }

    int dim = block_size;
    MultiDimRNN rnn(dim, 16, dim);

    std::cout << "[*] Training RNN on subset of blocks..." << std::endl;
    std::vector<std::vector<double>> train_data(sequences.begin(), sequences.begin() + std::min((size_t)100, sequences.size()));
    rnn.train(train_data, 50, 0.001);

    std::cout << "[*] Compressing and Verifying..." << std::endl;
    std::vector<double> h_enc(16, 0.0);
    std::vector<double> h_dec(16, 0.0);
    std::vector<int> prev_reconstructed;
    for(double d : sequences[0]) prev_reconstructed.push_back(static_cast<int>(d));

    int errors = 0;
    for (size_t i = 1; i < sequences.size(); ++i) {
        // Compress (Encode)
        std::vector<double> h_tmp = h_enc;
        std::vector<double> prediction = rnn.forward(sequences[i-1], h_enc);

        std::vector<int> residuals;
        for (int j = 0; j < dim; ++j) {
            int target = static_cast<int>(sequences[i][j]);
            int pred_val = static_cast<int>(std::round(prediction[j]));
            residuals.push_back(target - pred_val);
        }
        FiniteFieldVector compressed_vec(residuals);

        // Decompress (Decode)
        std::vector<double> prev_d;
        for(int v : prev_reconstructed) prev_d.push_back(static_cast<double>(v));
        std::vector<double> dec_prediction = rnn.forward(prev_d, h_dec);

        std::vector<int> current_reconstructed;
        for (int j = 0; j < dim; ++j) {
            int ff_val = compressed_vec.elements[j].value;
            int residual = (ff_val > PRIME/2) ? (ff_val - PRIME) : ff_val;
            int decompressed_val = static_cast<int>(std::round(dec_prediction[j] + residual));

            if (decompressed_val != static_cast<int>(sequences[i][j])) errors++;
            current_reconstructed.push_back(decompressed_val);
        }
        prev_reconstructed = current_reconstructed;
    }

    if (errors == 0) {
        std::cout << "[SUCCESS] Perfect reconstruction of " << sequences.size() << " blocks!" << std::endl;
    } else {
        std::cout << "[FAILURE] " << errors << " errors found." << std::endl;
    }

    return 0;
}
