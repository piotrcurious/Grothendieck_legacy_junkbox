#include <stdexcept>
#include <cstdint>
#include "rnn_absolute_core.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <map>
#include <iomanip>

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

    auto skipComments = [](std::ifstream& f) {
        char ch;
        while (f >> std::ws && f.peek() == '#') f.ignore(10000, '\n');
    };

    skipComments(file);
    int w, h;
    file >> w >> h;
    skipComments(file);
    int maxVal;
    file >> maxVal;
    file.ignore(1);

    Image img;
    img.width = w; img.height = h;
    size_t total_pixels = (size_t)w * h;
    img.data.resize(total_pixels);
    file.read(reinterpret_cast<char*>(img.data.data()), total_pixels);
    return img;
}

void savePGM(const std::string& filename, const Image& img) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Could not open output file: " + filename);
    file << "P5\n" << img.width << " " << img.height << "\n255\n";
    file.write(reinterpret_cast<const char*>(img.data.data()), img.data.size());
}

double calculateEntropy(const std::vector<int>& data) {
    if (data.empty()) return 0;
    std::map<int, int> counts;
    for (int x : data) counts[x]++;
    double entropy = 0;
    double size = data.size();
    for (auto const& [val, count] : counts) {
        double p = (double)count / size;
        entropy -= p * std::log2(p);
    }
    return entropy;
}

struct CompressionMetrics {
    std::string name;
    long original_size;
    double original_entropy;
    double rnn_residual_entropy;
    double delta_entropy;
    bool success;
};

CompressionMetrics processImage(const std::string& input_path) {
    std::cout << "--- Processing: " << input_path << " ---" << std::endl;
    Image img = loadPGM(input_path);
    const int block_size = 8;
    std::vector<std::vector<double>> sequences;
    for (size_t i = 0; i + block_size <= img.data.size(); i += block_size) {
        std::vector<double> seq;
        for (int j = 0; j < block_size; ++j) seq.push_back(static_cast<double>(img.data[i+j]) / 255.0);
        sequences.push_back(seq);
    }

    int dim = block_size;
    int hidden = 16;
    MultiDimRNN rnn(dim, hidden, dim);

    std::cout << "[*] Training RNN..." << std::endl;
    size_t train_size = std::min((size_t)1000, sequences.size());
    std::vector<std::vector<double>> train_data(sequences.begin(), sequences.begin() + train_size);
    rnn.train(train_data, 50, 0.01);

    std::cout << "[*] Running Compression..." << std::endl;
    std::vector<double> h_enc(hidden, 0.0);
    std::vector<double> h_dec(hidden, 0.0);
    Image reconstructed;
    reconstructed.width = img.width; reconstructed.height = img.height;
    reconstructed.data.assign(img.data.size(), 0);

    std::vector<int> rnn_residuals;
    std::vector<int> delta_residuals;
    std::vector<uint8_t> residual_view;
    residual_view.assign(img.data.size(), 128);

    for(int j=0; j<block_size; ++j) reconstructed.data[j] = img.data[j];

    int errors = 0;
    for (size_t i = 1; i < sequences.size(); ++i) {
        std::vector<double> prediction = rnn.forward(sequences[i-1], h_enc);
        for (int j = 0; j < dim; ++j) {
            int target = (int)img.data[i*block_size + j];
            int pred_val = (int)std::round(prediction[j] * 255.0);
            int res = target - pred_val;
            rnn_residuals.push_back(res);

            int delta_pred = (int)img.data[(i-1)*block_size + j];
            delta_residuals.push_back(target - delta_pred);

            int view_val = res + 128;
            if (view_val < 0) view_val = 0;
            if (view_val > 255) view_val = 255;
            residual_view[i*block_size + j] = (uint8_t)view_val;
        }

        std::vector<double> prev_d;
        for(int j=0; j<dim; ++j) prev_d.push_back((double)reconstructed.data[(i-1)*block_size + j] / 255.0);
        std::vector<double> dec_prediction = rnn.forward(prev_d, h_dec);
        for (int j = 0; j < dim; ++j) {
            int res = rnn_residuals[(i-1)*dim + j];
            FiniteFieldElement fe(res);
            int residual = (fe.value > PRIME/2) ? (fe.value - PRIME) : fe.value;
            int decompressed_val = (int)std::round(dec_prediction[j] * 255.0 + residual);
            if (decompressed_val < 0) decompressed_val = 0;
            if (decompressed_val > 255) decompressed_val = 255;
            if (decompressed_val != (int)img.data[i*block_size + j]) errors++;
            reconstructed.data[i*block_size + j] = (uint8_t)decompressed_val;
        }
    }

    double original_ent = calculateEntropy(std::vector<int>(img.data.begin(), img.data.end()));
    double rnn_ent = calculateEntropy(rnn_residuals);
    double delta_ent = calculateEntropy(delta_residuals);

    std::string base = input_path.substr(input_path.find_last_of("/\\") + 1);
    savePGM("rnn_absolute/reconstructed_" + base, reconstructed);
    savePGM("rnn_absolute/residuals_" + base, {img.width, img.height, residual_view});

    return {base, (long)img.data.size(), original_ent, rnn_ent, delta_ent, (errors == 0)};
}

int main() {
    try {
        std::vector<std::string> images = {
            "absolute_galois_group/compressor/01/test.pgm",
            "absolute_galois_group/compressor/01/GhostInShell_02_005.pgm"
        };

        std::vector<CompressionMetrics> results;
        for (const auto& img : images) {
            results.push_back(processImage(img));
        }

        std::ofstream report("rnn_absolute/compression_report.md");
        report << "# RNN Absolute Galois Group Compression Report\n\n";
        report << "| Image | Original Size | Orig. Entropy | Delta Entropy | RNN Res. Entropy | Ratio (Theoretical) | Reconstruction |\n";
        report << "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n";

        for (const auto& r : results) {
            double ratio = r.original_entropy / r.rnn_residual_entropy;
            report << "| " << r.name << " | " << r.original_size << " B | "
                   << std::fixed << std::setprecision(4) << r.original_entropy << " bpp | "
                   << r.delta_entropy << " bpp | "
                   << r.rnn_residual_entropy << " bpp | "
                   << ratio << ":1 | "
                   << (r.success ? "SUCCESS" : "FAILURE") << " |\n";
        }

        std::cout << "[*] Report generated: rnn_absolute/compression_report.md" << std::endl;

    } catch (std::exception& e) {
        std::cerr << "Fatal Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
