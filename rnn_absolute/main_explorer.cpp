#include <stdexcept>
#include <cstdint>
#include "rnn_absolute_core.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <map>
#include <iomanip>
#include <algorithm>

struct Image {
    int width, height;
    std::vector<uint8_t> data;
};

struct Metrics {
    std::string name;
    double orig_ent;
    double rnn_ent;
    double lzma_ratio;
    bool success;
};

Image loadPGM(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Could not open input file: " + filename);
    std::string magic; file >> magic;
    auto skip = [](std::ifstream& f) { char ch; while (f >> std::ws && f.peek() == '#') f.ignore(10000, '\n'); };
    skip(file); int w, h; file >> w >> h;
    skip(file); int maxVal; file >> maxVal;
    file.ignore(1);
    Image img; img.width = w; img.height = h;
    img.data.resize((size_t)w * h);
    file.read(reinterpret_cast<char*>(img.data.data()), img.data.size());
    return img;
}

double calculateEntropy(const std::vector<int>& data) {
    if (data.empty()) return 0;
    std::map<int, long long> counts;
    for (int x : data) counts[x]++;
    double entropy = 0, size = data.size();
    for (auto const& [val, count] : counts) {
        double p = (double)count / size;
        entropy -= p * std::log2(p);
    }
    return entropy;
}

double getMED(double a, double b, double c) {
    if (c >= std::max(a, b)) return std::min(a, b);
    if (c <= std::min(a, b)) return std::max(a, b);
    return a + b - c;
}

Metrics processImageHybrid(const std::string& path, double lr, int hid) {
    std::cout << "[*] Processing Hybrid (Galois + RNN + LZMA): " << path << std::endl;
    Image img = loadPGM(path);
    int w = img.width, h_img = img.height;

    int ctx_size = 29; GatedRNN predictor(ctx_size, hid);

    std::vector<uint8_t> orbits;
    std::vector<uint8_t> conjugacy_indices;
    std::vector<uint8_t> reconstructed_data(w * h_img, 0);
    int errs = 0;

    for(int y=0; y<h_img; ++y) {
        for(int x=0; x<w; ++x) {
            int i = y * w + x;
            if (y < 8 || x < 8 || x >= w - 8) {
                reconstructed_data[i] = img.data[i];
                orbits.push_back(img.data[i]);
                conjugacy_indices.push_back(0);
                continue;
            }
            auto getV = [&](int tx, int ty) { return ((double)reconstructed_data[ty * w + tx] - 128.0) / 128.0; };
            std::vector<double> input = {
                getV(x-1, y), getV(x-2, y), getV(x-3, y), getV(x, y-1), getV(x, y-2), getV(x, y-3),
                getV(x-1, y-1), getV(x+1, y-1), getV(x-1, y-2), getV(x+1, y-2), getV(x-2, y-1), getV(x+2, y-1),
                getMED(getV(x-1, y), getV(x, y-1), getV(x-1, y-1)), getV(x-1, y) + getV(x, y-1) - getV(x-1, y-1), (getV(x-1, y) + getV(x, y-1)) / 2.0,
                (double)x/w, (double)y/h_img
            };
            int scales[] = {2, 4, 8};
            for(int s : scales) {
                input.push_back(getV(x-s, y)); input.push_back(getV(x, y-s)); input.push_back(getV(x-s, y-s)); input.push_back(getV(x+s, y-s));
            }

            double pred = predictor.forward(input);
            int target = (int)img.data[i];
            int p_val = (int)std::round(pred * 128.0 + 128.0);

            uint8_t res_byte = (uint8_t)(target - p_val);
            GaloisOrbit g_res = get_canonical({res_byte});

            orbits.push_back(g_res.canonical[0]);
            conjugacy_indices.push_back((uint8_t)g_res.k);

            uint8_t recovered_res = frobenius(g_res.canonical[0], (8 - g_res.k) % 8);
            reconstructed_data[i] = (uint8_t)(p_val + (int8_t)recovered_res);

            if (reconstructed_data[i] != target) errs++;
            predictor.train(input, ((double)target - 128.0) / 128.0, lr);
        }
    }

    std::vector<uint8_t> stream;
    stream.insert(stream.end(), orbits.begin(), orbits.end());
    stream.insert(stream.end(), conjugacy_indices.begin(), conjugacy_indices.end());
    auto compressed = lzma_compress(stream);
    double lz_ratio = (double)img.data.size() / compressed.size();

    std::vector<int> orbit_ints; for(auto x : orbits) orbit_ints.push_back((int)x);

    return {path, calculateEntropy(std::vector<int>(img.data.begin(), img.data.end())), calculateEntropy(orbit_ints), lz_ratio, errs == 0};
}

int main() {
    std::string img1 = "../absolute_galois_group/compressor/01/test.pgm";
    std::string img2 = "../absolute_galois_group/compressor/01/GhostInShell_02_005.pgm";

    auto m1 = processImageHybrid(img1, 0.012, 48);
    auto m2 = processImageHybrid(img2, 0.012, 48);

    std::ofstream report("compression_report.md");
    report << "# Absolute Galois RNN + LZMA Compression Report\n\n";
    report << "| Image | Orig Entropy | Orbit Entropy | LZMA Ratio | Recon |\n";
    report << "| :--- | :--- | :--- | :--- | :--- |\n";
    auto add = [&](Metrics m) {
        report << "| " << m.name << " | " << std::fixed << std::setprecision(4) << m.orig_ent << " | " << m.rnn_ent << " | " << m.lzma_ratio << ":1 | " << (m.success ? "SUCCESS" : "FAIL") << " |\n";
    };
    add(m1); add(m2);
    return 0;
}
