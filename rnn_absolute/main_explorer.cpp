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

struct Point { int x, y; };

void quadtree_traverse(int x, int y, int size, int w, int h, std::vector<Point>& out) {
    if (x >= w || y >= h) return;
    if (size <= 8) {
        for(int j=y; j<std::min(y+size, h); ++j) for(int i=x; i<std::min(x+size, w); ++i) out.push_back({i, j});
        return;
    }
    int hs = size / 2;
    quadtree_traverse(x, y, hs, w, h, out);
    quadtree_traverse(x + hs, y, hs, w, h, out);
    quadtree_traverse(x, y + hs, hs, w, h, out);
    quadtree_traverse(x + hs, y + hs, hs, w, h, out);
}

Metrics processImageFractalManifold(const std::string& path, double lr, int hid) {
    std::cout << "[*] Processing Fractal Manifold RNN (Robust): " << path << std::endl;
    Image img = loadPGM(path);
    int w = img.width, h_img = img.height;

    int qs = 1; while(qs < std::max(w, h_img)) qs *= 2;
    std::vector<Point> order;
    quadtree_traverse(0, 0, qs, w, h_img, order);

    // Feature vector: 4 neighbors * 6 manifold features + 2 pos + 4 multi-scale = 24 + 2 + 4 = 30
    int ctx_size = 30;
    GaloisRNN predictor(ctx_size, hid, (int)GF8.orbits.size());

    std::vector<uint8_t> orbit_stream;
    std::vector<uint8_t> root_stream;
    std::vector<uint8_t> reconstructed_data(w * h_img, 0);
    std::vector<bool> visited(w * h_img, false);
    int errs = 0;

    for(const auto& p : order) {
        int i = p.y * w + p.x;
        // Increased safety border for fractal offsets
        bool is_border = (p.y < 16 || p.x < 16 || p.x >= w - 16 || p.y >= h_img - 16);
        if (is_border) {
            reconstructed_data[i] = img.data[i];
            orbit_stream.push_back((uint8_t)GF8.element_to_orbit_id[img.data[i]]);
            root_stream.push_back((uint8_t)GF8.element_to_root_index[img.data[i]]);
            visited[i] = true;
            continue;
        }

        auto getV = [&](int tx, int ty) {
            if (tx < 0 || tx >= w || ty < 0 || ty >= h_img || !visited[ty * w + tx]) return (uint8_t)0;
            return reconstructed_data[ty * w + tx];
        };
        std::vector<double> input;

        int nx[] = {p.x-1, p.x, p.x-1, p.x+1}, ny[] = {p.y, p.y-1, p.y-1, p.y-1};
        for(int j=0; j<4; ++j) {
            uint8_t v = getV(nx[j], ny[j]);
            input.push_back((double)GF8.tr8_1(v));
            input.push_back((double)GF8.tr8_4(v)/255.0);
            input.push_back((double)GF8.element_to_orbit_id[v]/(double)GF8.orbits.size());
            input.push_back((double)GF8.orbits[GF8.element_to_orbit_id[v]].degree/8.0);
            input.push_back((double)v/255.0);
            input.push_back(std::abs((double)v - (double)getV(p.x-2,p.y))/255.0);
        }
        input.push_back((double)p.x/w); input.push_back((double)p.y/h_img);
        int scales[] = {2, 4, 8, 16};
        for(int s : scales) input.push_back((double)getV(p.x-s, p.y)/255.0);

        // Verification of input size alignment
        if(input.size() != (size_t)ctx_size) {
             throw std::runtime_error("Context size mismatch: expected " + std::to_string(ctx_size) + " got " + std::to_string(input.size()));
        }

        auto probs = predictor.forward(input).first;
        int actual_orbit = GF8.element_to_orbit_id[img.data[i]];
        int actual_root = GF8.element_to_root_index[img.data[i]];

        orbit_stream.push_back((uint8_t)actual_orbit);
        root_stream.push_back((uint8_t)actual_root);

        reconstructed_data[i] = GF8.orbits[actual_orbit].elements[actual_root];
        if (reconstructed_data[i] != img.data[i]) errs++;
        visited[i] = true;

        predictor.train(input, actual_orbit, actual_root, lr);
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
    std::string img1 = "../absolute_galois_group/compressor/01/test.pgm";
    std::string img2 = "../absolute_galois_group/compressor/01/GhostInShell_02_005.pgm";

    auto m1 = processImageFractalManifold(img1, 0.012, 64);
    auto m2 = processImageFractalManifold(img2, 0.012, 64);

    std::ofstream report("compression_report.md");
    report << "# Galois Fractal Manifold Compression Report (Robust Build)\n\n";
    report << "| Image | Orig Entropy | Orbit Entropy | LZMA Ratio | Recon |\n";
    report << "| :--- | :--- | :--- | :--- | :--- |\n";
    auto add = [&](Metrics m) {
        report << "| " << m.name << " | " << std::fixed << std::setprecision(4) << m.orig_ent << " | " << m.orbit_ent << " | " << m.lzma_ratio << ":1 | " << (m.success ? "SUCCESS" : "FAIL") << " |\n";
    };
    add(m1); add(m2);
    return 0;
}
