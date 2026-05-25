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
    bool success;
};

Image loadPGM(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Could not open input file");
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
    double entropy = 0, size = data.size();
    for (auto const& [val, count] : counts) {
        double p = (double)count / size;
        entropy -= p * std::log2(p);
    }
    return entropy;
}

struct Point { uint32_t x, y, z; };

double getMED(double a, double b, double c) {
    if (c >= std::max(a, b)) return std::min(a, b);
    if (c <= std::min(a, b)) return std::max(a, b);
    return a + b - c;
}

Metrics processImageRaster(const std::string& path, double lr, int hid) {
    std::cout << "[*] Processing Raster: " << path << std::endl;
    Image img = loadPGM(path);
    int w = img.width, h_img = img.height;
    int ctx_size = 12 + 1 + 2;
    GatedRNN predictor(ctx_size, hid);
    std::vector<int> rnn_residuals;
    std::vector<uint8_t> reconstructed_data(w * h_img, 0);

    int errs = 0;
    for(int y=0; y<h_img; ++y) {
        for(int x=0; x<w; ++x) {
            int i = y * w + x;
            if (y < 3 || x < 3 || x >= w - 3) {
                reconstructed_data[i] = img.data[i];
                rnn_residuals.push_back(0);
                continue;
            }
            auto getV = [&](int tx, int ty) { return ((double)reconstructed_data[ty * w + tx] - 128.0) / 128.0; };
            std::vector<double> input = {
                getV(x-1, y), getV(x-2, y), getV(x-3, y),
                getV(x, y-1), getV(x, y-2), getV(x, y-3),
                getV(x-1, y-1), getV(x+1, y-1),
                getV(x-1, y-2), getV(x+1, y-2),
                getV(x-2, y-1), getV(x+2, y-1),
                getMED(getV(x-1, y), getV(x, y-1), getV(x-1, y-1)),
                (double)x/w, (double)y/h_img
            };
            double pred = predictor.forward(input);
            int target = (int)img.data[i];
            int p_val = (int)std::round(pred * 128.0 + 128.0);
            FiniteFieldElement fe(target - p_val);
            int ff_res = (fe.value > PRIME/2) ? (fe.value - PRIME) : fe.value;
            rnn_residuals.push_back(ff_res);
            int recon_val = p_val + ff_res;
            if (recon_val < 0) recon_val = 0; if (recon_val > 255) recon_val = 255;
            reconstructed_data[i] = (uint8_t)recon_val;
            if (reconstructed_data[i] != target) errs++;
            predictor.train(input, ((double)target - 128.0) / 128.0, lr);
        }
    }
    double o_e = calculateEntropy(std::vector<int>(img.data.begin(), img.data.end()));
    double r_e = calculateEntropy(rnn_residuals);
    std::string base = path.substr(path.find_last_of("/\\") + 1);
    savePGM("rnn_absolute/reconstructed_" + base, {w, h_img, reconstructed_data});
    return {base, o_e, r_e, errs == 0};
}

Metrics processImageMorton(const std::string& path, double lr, int hid) {
    std::cout << "[*] Processing Morton: " << path << std::endl;
    Image img = loadPGM(path);
    int w = img.width, h_img = img.height;
    std::vector<Point> points;
    for(uint32_t y=0; y<h_img; ++y) for(uint32_t x=0; x<w; ++x) points.push_back({x, y, morton2D(x, y)});
    std::sort(points.begin(), points.end(), [](const Point& a, const Point& b) { return a.z < b.z; });

    int ctx_size = 12 + 1 + 2;
    GatedRNN predictor(ctx_size, hid);
    std::vector<int> rnn_residuals;
    std::vector<uint8_t> reconstructed_data(w * h_img, 0);
    std::vector<bool> visited(w * h_img, false);

    int errs = 0;
    for(const auto& p : points) {
        auto getV = [&](int tx, int ty) {
            if (tx < 0 || tx >= w || ty < 0 || ty >= h_img || !visited[ty * w + tx]) return 0.0;
            return ((double)reconstructed_data[ty * w + tx] - 128.0) / 128.0;
        };
        std::vector<double> input = {
            getV(p.x-1, p.y), getV(p.x-2, p.y), getV(p.x-3, p.y),
            getV(p.x, p.y-1), getV(p.x, p.y-2), getV(p.x, p.y-3),
            getV(p.x-1, p.y-1), getV(p.x+1, p.y-1),
            getV(p.x-1, p.y-2), getV(p.x+1, p.y-2),
            getV(p.x-2, p.y-1), getV(p.x+2, p.y-1),
            getMED(getV(p.x-1, p.y), getV(p.x, p.y-1), getV(p.x-1, p.y-1)),
            (double)p.x/w, (double)p.y/h_img
        };
        double pred = predictor.forward(input);
        int target = (int)img.data[p.y * w + p.x];
        int p_val = (int)std::round(pred * 128.0 + 128.0);
        FiniteFieldElement fe(target - p_val);
        int ff_res = (fe.value > PRIME/2) ? (fe.value - PRIME) : fe.value;
        rnn_residuals.push_back(ff_res);
        int recon_val = p_val + ff_res;
        if (recon_val < 0) recon_val = 0; if (recon_val > 255) recon_val = 255;
        reconstructed_data[p.y * w + p.x] = (uint8_t)recon_val;
        if (reconstructed_data[p.y * w + p.x] != target) errs++;
        visited[p.y * w + p.x] = true;
        predictor.train(input, ((double)target - 128.0) / 128.0, lr);
    }
    double o_e = calculateEntropy(std::vector<int>(img.data.begin(), img.data.end()));
    double r_e = calculateEntropy(rnn_residuals);
    std::string base = path.substr(path.find_last_of("/\\") + 1);
    savePGM("rnn_absolute/morton_reconstructed_" + base, {w, h_img, reconstructed_data});
    return {base, o_e, r_e, errs == 0};
}

int main() {
    auto r1 = processImageRaster("absolute_galois_group/compressor/01/test.pgm", 0.01, 32);
    auto m1 = processImageMorton("absolute_galois_group/compressor/01/test.pgm", 0.01, 32);
    auto r2 = processImageRaster("absolute_galois_group/compressor/01/GhostInShell_02_005.pgm", 0.01, 32);
    auto m2 = processImageMorton("absolute_galois_group/compressor/01/GhostInShell_02_005.pgm", 0.01, 32);

    std::ofstream report("rnn_absolute/compression_report.md");
    report << "# RNN Compression Comparison Report (Optimized GatedRNN + MED + Positional)\n\n";
    report << "| Image | Mode | Original Entropy | RNN Entropy | Ratio | Recon |\n";
    report << "| :--- | :--- | :--- | :--- | :--- | :--- |\n";
    auto add = [&](Metrics m, std::string mode) {
        report << "| " << m.name << " | " << mode << " | " << std::fixed << std::setprecision(4) << m.orig_ent << " | " << m.rnn_ent << " | " << m.orig_ent/m.rnn_ent << " | " << (m.success ? "SUCCESS" : "FAIL") << " |\n";
    };
    add(r1, "Raster"); add(m1, "Morton");
    add(r2, "Raster"); add(m2, "Morton");
    return 0;
}
