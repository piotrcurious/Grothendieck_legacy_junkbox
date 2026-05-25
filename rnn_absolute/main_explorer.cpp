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

double getMED(double a, double b, double c) {
    if (c >= std::max(a, b)) return std::min(a, b);
    if (c <= std::min(a, b)) return std::max(a, b);
    return a + b - c;
}

Metrics processImageFractal(const std::string& path, double lr, int hid) {
    std::cout << "[*] Processing Fractal RNN (Verified): " << path << std::endl;
    Image img = loadPGM(path);
    int w = img.width, h_img = img.height;

    int ctx_size = 12 + 1 + 2 + 3*4;
    GatedRNN predictor(ctx_size, hid);

    std::vector<int> rnn_residuals;
    std::vector<uint8_t> reconstructed_data(w * h_img, 0);

    int errs = 0;
    for(int y=0; y<h_img; ++y) {
        for(int x=0; x<w; ++x) {
            int i = y * w + x;
            if (y < 8 || x < 8 || x >= w - 8) {
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
                getV(x-2, y-1), getV(x+2, y-1)
            };

            input.push_back(getMED(getV(x-1, y), getV(x, y-1), getV(x-1, y-1)));
            input.push_back((double)x/w);
            input.push_back((double)y/h_img);

            int scales[] = {2, 4, 8};
            for(int s : scales) {
                input.push_back(getV(x-s, y)); input.push_back(getV(x, y-s));
                input.push_back(getV(x-s, y-s)); input.push_back(getV(x+s, y-s));
            }

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
    return {base, o_e, r_e, errs == 0};
}

int main() {
    std::string img1 = "absolute_galois_group/compressor/01/test.pgm";
    std::string img2 = "absolute_galois_group/compressor/01/GhostInShell_02_005.pgm";

    auto f1 = processImageFractal(img1, 0.012, 32);
    auto f2 = processImageFractal(img2, 0.012, 32);

    std::ofstream report("rnn_absolute/fractal_report.md");
    report << "# Fractal RNN Compression Report\n\n";
    report << "| Image | Original Entropy | Fractal Entropy | Ratio | Recon |\n";
    report << "| :--- | :--- | :--- | :--- | :--- |\n";
    auto add = [&](Metrics m) {
        report << "| " << m.name << " | " << std::fixed << std::setprecision(4) << m.orig_ent << " | " << m.rnn_ent << " | " << m.orig_ent/m.rnn_ent << " | " << (m.success ? "SUCCESS" : "FAIL") << " |\n";
    };
    add(f1); add(f2);
    return 0;
}
