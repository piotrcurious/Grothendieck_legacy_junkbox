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

Image loadPGM(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Could not open input file");
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

void savePGM(const std::string& filename, const Image& img) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Could not open output file");
    file << "P5\n" << img.width << " " << img.height << "\n255\n";
    file.write(reinterpret_cast<const char*>(img.data.data()), img.data.size());
}

double calculateEntropy(const std::vector<int>& data) {
    if (data.empty()) return 0;
    std::map<int, long long> counts;
    for (int x : data) counts[x]++;
    double entropy = 0;
    double size = data.size();
    for (auto const& [val, count] : counts) {
        double p = (double)count / size;
        entropy -= p * std::log2(p);
    }
    return entropy;
}

struct Metrics {
    std::string name;
    double orig_ent;
    double rnn_ent;
    bool success;
};

Metrics processImage(const std::string& path) {
    std::cout << "[*] Processing: " << path << " (OptimizedRNN + Adam + Context12)" << std::endl;
    Image img = loadPGM(path);
    int w = img.width;
    int h_img = img.height;

    int hidden = 16;
    int ctx_size = 12;
    OptimizedRNN predictor(ctx_size, hidden);

    std::vector<int> rnn_residuals;
    std::vector<uint8_t> res_view; res_view.assign(img.data.size(), 128);
    Image reconstructed; reconstructed.width = w; reconstructed.height = h_img; reconstructed.data.assign(img.data.size(), 0);

    double lr = 0.005;
    int errors = 0;

    for(int y = 0; y < h_img; ++y) {
        for(int x = 0; x < w; ++x) {
            int i = y * w + x;
            if (y < 3 || x < 3 || x >= w - 3) {
                reconstructed.data[i] = img.data[i];
                rnn_residuals.push_back(0);
                continue;
            }

            auto getV = [&](int tx, int ty) { return ((double)reconstructed.data[ty * w + tx] - 128.0) / 128.0; };

            std::vector<double> input = {
                getV(x-1, y), getV(x-2, y), getV(x-3, y),
                getV(x, y-1), getV(x, y-2), getV(x, y-3),
                getV(x-1, y-1), getV(x+1, y-1),
                getV(x-1, y-2), getV(x+1, y-2),
                getV(x-2, y-1), getV(x+2, y-1)
            };

            double pred = predictor.forward(input);

            int target = (int)img.data[i];
            int p_val = (int)std::round(pred * 128.0 + 128.0);
            int res = target - p_val;

            FiniteFieldElement fe(res);
            int ff_res = (fe.value > PRIME/2) ? (fe.value - PRIME) : fe.value;
            rnn_residuals.push_back(ff_res);

            int recon_val = p_val + ff_res;
            if (recon_val < 0) recon_val = 0; if (recon_val > 255) recon_val = 255;
            reconstructed.data[i] = (uint8_t)recon_val;

            if (reconstructed.data[i] != target) errors++;

            double nt = ((double)target - 128.0) / 128.0;
            predictor.train(input, nt, lr);

            int vv = ff_res + 128;
            if (vv < 0) vv = 0; if (vv > 255) vv = 255;
            res_view[i] = (uint8_t)vv;
        }
        if (y % 400 == 0) std::cout << "  Row: " << y << "/" << h_img << std::endl;
    }

    double o_e = calculateEntropy(std::vector<int>(img.data.begin(), img.data.end()));
    double r_e = calculateEntropy(rnn_residuals);

    std::string base = path.substr(path.find_last_of("/\\") + 1);
    savePGM("rnn_absolute/residuals_" + base, {w, h_img, res_view});
    savePGM("rnn_absolute/reconstructed_" + base, reconstructed);

    return {base, o_e, r_e, (errors == 0)};
}

int main() {
    auto m1 = processImage("absolute_galois_group/compressor/01/test.pgm");
    auto m2 = processImage("absolute_galois_group/compressor/01/GhostInShell_02_005.pgm");
    std::ofstream report("rnn_absolute/compression_report.md");
    report << "# RNN Absolute Galois Group Compression Report (OptimizedRNN + Adam + Context12)\n\n";
    report << "| Image | Original Entropy | RNN Entropy | Ratio | Recon |\n";
    report << "| :--- | :--- | :--- | :--- | :--- |\n";
    auto add = [&](Metrics m) {
        report << "| " << m.name << " | " << std::fixed << std::setprecision(4) << m.orig_ent << " | " << m.rnn_ent << " | " << m.orig_ent/m.rnn_ent << ":1 | " << (m.success ? "SUCCESS" : "FAIL") << " |\n";
    };
    add(m1); add(m2);
    return 0;
}
