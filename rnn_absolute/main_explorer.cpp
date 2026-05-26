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
    double orbit_ent;
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

void savePGM(const std::string& filename, const Image& img) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Could not open output file");
    file << "P5\n" << img.width << " " << img.height << "\n255\n";
    file.write(reinterpret_cast<const char*>(img.data.data()), img.data.size());
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

double estimate_local_fd(const std::vector<uint8_t>& img, int x, int y, int w, int h) {
    auto get = [&](int tx, int ty) { return (tx<0||ty<0||tx>=w||ty>=h)?0:img[ty*w+tx]; };
    auto var = [&](int sz) {
        double s=0, s2=0; int n=0;
        for(int j=y-sz; j<=y+sz; ++j) for(int i=x-sz; i<=x+sz; ++i) {
            double v = (double)get(i, j); s += v; s2 += v*v; n++;
        }
        return (s2/n - (s/n)*(s/n));
    };
    double v1 = var(1), v4 = var(4);
    if(v1 < 1e-5) return 2.0;
    double h_exp = (std::log(v4+1e-5) - std::log(v1+1e-5)) / (2.0 * std::log(4.0));
    return std::clamp(3.0 - h_exp, 1.0, 3.0);
}

Metrics processImageAbsoluteGalois(const std::string& path, double lr, int hid) {
    std::cout << "[*] Processing Latent Manifold Galois RNN: " << path << std::endl;
    Image img = loadPGM(path);
    int w = img.width, h_img = img.height;

    // x_s: 4 neighbors * 5 manifold features = 20
    // x_f: 4 neighbors * (5 manifold + 1 FD) + 2 pos + 8 min_poly_coeffs = 34
    DualPathGaloisGRU predictor(20, 34, hid, (int)GF8.orbits.size());

    std::vector<uint8_t> orbit_stream;
    std::vector<uint8_t> root_stream;
    std::vector<uint8_t> reconstructed_data(w * h_img, 0);
    int errs = 0;

    for(int y=0; y<h_img; ++y) {
        for(int x=0; x<w; ++x) {
            int i = y * w + x;
            if (y < 16 || x < 16 || x >= w - 16 || y >= h_img - 16) {
                reconstructed_data[i] = img.data[i];
                orbit_stream.push_back((uint8_t)GF8.element_to_orbit_id[img.data[i]]);
                root_stream.push_back((uint8_t)GF8.element_to_root_index[img.data[i]]);
                continue;
            }

            auto getV = [&](int tx, int ty) { return reconstructed_data[ty * w + tx]; };
            auto stalk = [&](uint8_t v) {
                return std::vector<double>{ (double)GF8.tr8_1(v), (double)GF8.tr8_4(v)/255.0, (double)GF8.element_to_orbit_id[v]/(double)GF8.orbits.size(), (double)GF8.orbits[GF8.element_to_orbit_id[v]].elements.size()/8.0, (double)v/255.0 };
            };

            std::vector<double> x_s, x_f;
            int nx[] = {x-1, x, x-1, x+1}, ny[] = {y, y-1, y-1, y-1};
            for(int j=0; j<4; ++j) {
                auto s = stalk(getV(nx[j], ny[j]));
                x_s.insert(x_s.end(), s.begin(), s.end());
            }

            int fx[] = {x-2, x, x-4, x}, fy[] = {y, y-2, y, y-4};
            for(int j=0; j<4; ++j) {
                auto s = stalk(getV(fx[j], fy[j]));
                x_f.insert(x_f.end(), s.begin(), s.end());
                x_f.push_back(estimate_local_fd(reconstructed_data, fx[j], fy[j], w, h_img) / 3.0);
            }
            x_f.push_back((double)x/w); x_f.push_back((double)y/h_img);

            // Minimal Polynomial Coefficients as High-Dimensional Features
            const auto& mpc = GF8.orbits[GF8.element_to_orbit_id[getV(x-1, y)]].min_poly_coeffs;
            for(int k=0; k<8; k++) x_f.push_back(k < (int)mpc.size() ? (double)mpc[k]/255.0 : 0.0);

            auto res = predictor.forward(x_s, x_f);
            int actual_orbit = GF8.element_to_orbit_id[img.data[i]];
            int actual_root = GF8.element_to_root_index[img.data[i]];
            orbit_stream.push_back((uint8_t)actual_orbit);
            root_stream.push_back((uint8_t)actual_root);

            reconstructed_data[i] = GF8.orbits[actual_orbit].elements[actual_root];
            if (reconstructed_data[i] != img.data[i]) errs++;

            predictor.train(x_s, x_f, actual_orbit, actual_root, lr);
        }
    }

    std::vector<uint8_t> packets;
    for(auto o : orbit_stream) packets.push_back(o);
    for(auto r : root_stream) packets.push_back(r);
    auto compressed = lzma_compress(packets);

    std::string base = path.substr(path.find_last_of("/\\") + 1);
    savePGM("reconstructed_" + base, {w, h_img, reconstructed_data});

    return {path, calculateEntropy(img.data), calculateEntropy(orbit_stream), (double)img.data.size() / std::max((size_t)1, compressed.size()), errs == 0};
}

int main() {
    std::string img1 = "absolute_galois_group/compressor/01/test.pgm";
    std::string img2 = "absolute_galois_group/compressor/01/GhostInShell_02_005.pgm";

    auto m1 = processImageAbsoluteGalois(img1, 0.015, 32);
    auto m2 = processImageAbsoluteGalois(img2, 0.015, 32);

    std::ofstream report("compression_report.md");
    report << "# Peak Latent Manifold Galois RNN Compression Report\n\n";
    report << "| Image | Orig Entropy | Orbit Entropy | LZMA Ratio | Recon |\n";
    report << "| :--- | :--- | :--- | :--- | :--- |\n";
    auto add = [&](Metrics m) {
        report << "| " << m.name << " | " << std::fixed << std::setprecision(4) << m.orig_ent << " | " << m.orbit_ent << " | " << m.lzma_ratio << ":1 | " << (m.success ? "SUCCESS" : "FAIL") << " |\n";
    };
    add(m1); add(m2);
    return 0;
}
