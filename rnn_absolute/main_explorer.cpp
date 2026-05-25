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
    double canonical_ent;
    double lzma_ratio;
    bool success;
};

Image loadPGM(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Could not open input file: " + filename);
    std::string magic; file >> magic;
    if (magic != "P5") throw std::runtime_error("Only P5 supported");
    auto skip = [](std::ifstream& f) { char ch; while (f >> std::ws && f.peek() == '#') f.ignore(10000, '\n'); };
    skip(file); int w, h; file >> w >> h;
    skip(file); int maxVal; file >> maxVal;
    file.ignore(1);
    Image img; img.width = w; img.height = h;
    img.data.resize((size_t)w * h);
    file.read(reinterpret_cast<char*>(img.data.data()), img.data.size());
    return img;
}

double calculateEntropy(const std::vector<uint8_t>& data) {
    if (data.empty()) return 0;
    std::map<uint8_t, long long> counts;
    for (uint8_t x : data) counts[x]++;
    double entropy = 0, size = data.size();
    for (auto const& [val, count] : counts) {
        double p = (double)count / size;
        entropy -= p * std::log2(p);
    }
    return entropy;
}

Metrics processImageTrueGalois(const std::string& path, double lr, int hid) {
    std::cout << "[*] Processing: " << path << " (True Absolute Galois Action)" << std::endl;
    Image img = loadPGM(path);
    int w = img.width, h_img = img.height;

    int ctx_size = 30; // 29 + 1 (Trace)
    GaloisGatedRNN predictor(ctx_size, hid);

    std::vector<uint8_t> canonical_stream;
    std::vector<uint8_t> k_stream;
    std::vector<uint8_t> reconstructed_data(w * h_img, 0);
    int errs = 0;

    for(int y=0; y<h_img; ++y) {
        for(int x=0; x<w; ++x) {
            int i = y * w + x;
            if (y < 8 || x < 8 || x >= w - 8) {
                reconstructed_data[i] = img.data[i];
                canonical_stream.push_back(img.data[i]);
                k_stream.push_back(0);
                continue;
            }
            auto getV = [&](int tx, int ty) { return ((double)reconstructed_data[ty * w + tx] - 128.0) / 128.0; };
            std::vector<double> input = {
                getV(x-1, y), getV(x-2, y), getV(x-3, y), getV(x, y-1), getV(x, y-2), getV(x, y-3),
                getV(x-1, y-1), getV(x+1, y-1), getV(x-1, y-2), getV(x+1, y-2), getV(x-2, y-1), getV(x+2, y-1)
            };
            double a = getV(x-1, y), b = getV(x, y-1), c = getV(x-1, y-1);
            input.push_back(std::max(std::min(a,b), std::min(std::max(a,b),c)));
            input.push_back(a + b - c);
            input.push_back((a + b) / 2.0);
            input.push_back((double)x/w); input.push_back((double)y/h_img);
            int scales[] = {2, 4, 8};
            for(int s : scales) {
                input.push_back(getV(x-s, y)); input.push_back(getV(x, y-s));
                input.push_back(getV(x-s, y-s)); input.push_back(getV(x+s, y-s));
            }
            // Add Field Trace of previous pixel as feature
            input.push_back((double)GF8.trace(reconstructed_data[i-1]));

            double pred = predictor.forward(input);
            int target = (int)img.data[i];
            int p_val = (int)std::round(pred * 128.0 + 128.0);

            uint8_t res_byte = (uint8_t)(target - p_val);
            int k;
            uint8_t canon = GF8.get_canonical(res_byte, k);

            canonical_stream.push_back(canon);
            k_stream.push_back((uint8_t)k);

            uint8_t recon_res = canon;
            for(int j=0; j<(8-k)%8; ++j) recon_res = GF8.frobenius(recon_res);
            reconstructed_data[i] = (uint8_t)(p_val + (int8_t)recon_res);

            if (reconstructed_data[i] != target) errs++;
            predictor.train(input, ((double)target - 128.0) / 128.0, lr);
        }
    }

    std::vector<uint8_t> packets;
    packets.insert(packets.end(), canonical_stream.begin(), canonical_stream.end());
    packets.insert(packets.end(), k_stream.begin(), k_stream.end());
    auto compressed = lzma_compress(packets);

    return {path, calculateEntropy(std::vector<uint8_t>(img.data.begin(), img.data.end())), calculateEntropy(canonical_stream), (double)img.data.size() / compressed.size(), errs == 0};
}

int main() {
    std::string img1 = "../absolute_galois_group/compressor/01/test.pgm";
    std::string img2 = "../absolute_galois_group/compressor/01/GhostInShell_02_005.pgm";

    auto m1 = processImageTrueGalois(img1, 0.012, 48);
    auto m2 = processImageTrueGalois(img2, 0.012, 48);

    std::ofstream report("compression_report.md");
    report << "# Absolute Galois Group RNN Compression (True Action + LZMA)\n\n";
    report << "| Image | Orig Entropy | Canonical Entropy | LZMA Ratio | Recon |\n";
    report << "| :--- | :--- | :--- | :--- | :--- |\n";
    auto add = [&](Metrics m) {
        report << "| " << m.name << " | " << std::fixed << std::setprecision(4) << m.orig_ent << " | " << m.canonical_ent << " | " << m.lzma_ratio << ":1 | " << (m.success ? "SUCCESS" : "FAIL") << " |\n";
    };
    add(m1); add(m2);
    return 0;
}
