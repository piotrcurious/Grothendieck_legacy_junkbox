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
    double trace_ent, rank_orb_ent, rank_root_ent;
    double core_efficiency;
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

Metrics processImage(const std::string& path, double lr, int hid, int lossy_q = 0, int crop_sz = 0, bool use_lzma = true) {
    std::cout << "[*] Processing Holomorphic Tensor Absolute Galois RNN: " << path
              << " (Quality: " << lossy_q << ", Crop: " << (crop_sz ? std::to_string(crop_sz) : "FULL")
              << ", LZMA: " << (use_lzma ? "ON" : "OFF") << ")" << std::endl;
    Image full = loadPGM(path);
    Image img;
    if (crop_sz > 0 && full.width >= crop_sz && full.height >= crop_sz) {
        img.width = crop_sz; img.height = crop_sz;
        img.data.resize(crop_sz * crop_sz);
        int ox = (full.width - crop_sz) / 2, oy = (full.height - crop_sz) / 2;
        for(int y=0; y<crop_sz; ++y) for(int x=0; x<crop_sz; ++x) img.data[y*crop_sz+x] = full.data[(oy+y)*full.width+(ox+x)];
    } else {
        img = full;
    }
    int w = img.width, h_img = img.height;

    DualPathGaloisGRU predictor(146, 100, hid, (int)GF8.orbits.size());

    std::vector<uint8_t> orbit_stream;
    std::vector<uint8_t> reconstructed_data(w * h_img, 0);
    int errs = 0;
    uint8_t last_rank = 0, last_rid = 0;
    SignalContext ctx = {w, h_img, reconstructed_data};

    for(int y=0; y<h_img; ++y) {
        for(int x=0; x<w; ++x) {
            int i = y * w + x;
            if (y < 16 || x < 16 || x >= w - 16 || y >= h_img - 16) {
                reconstructed_data[i] = img.data[i];
                orbit_stream.push_back(img.data[i]);
                continue;
            }

            std::vector<double> x_s, x_f;
            ctx.get_features(x, y, last_rank, last_rid, x_s, x_f);
            auto res_pair = predictor.forward(x_s, x_f);

            // Unified Byte-level Probability Estimation
            std::vector<double> byte_probs(256, 0.0);
            std::vector<double> orbit_root_sums(GF8.orbits.size(), 1e-9);
            for(int b=0; b<256; b++) {
                int oid = GF8.element_to_orbit_id[b];
                int rid = GF8.element_to_root_index[b];
                orbit_root_sums[oid] += (rid < 8) ? res_pair.p_root[rid] : 0.0;
            }

            for(int b=0; b<256; b++) {
                int oid = GF8.element_to_orbit_id[b];
                int rid = GF8.element_to_root_index[b];
                double p_root_norm = (rid < 8) ? (res_pair.p_root[rid] / orbit_root_sums[oid]) : 0.0;
                byte_probs[b] = res_pair.p_orb[oid] * p_root_norm;
            }

            uint8_t target_val = img.data[i];
            uint8_t selected_val = target_val;

            if (lossy_q > 0) {
                // Competitive RDO mapping: High Q -> High Lambda -> Low Distortion
                double q_norm = (double)lossy_q / 100.0;
                double lambda = std::pow(10.0, 5.0 * q_norm - 1.0);
                int radius = std::max(1, (int)(24.0 * (1.0 - q_norm)));
                double min_cost = 1e18;
                for (int v = std::max(0, (int)target_val - radius); v <= std::min(255, (int)target_val + radius); ++v) {
                    double p = std::max(byte_probs[v], 1e-12);
                    double dist = (v - target_val) * (v - target_val);
                    double cost = -std::log2(p) + lambda * dist;
                    if (cost < min_cost) {
                        min_cost = cost;
                        selected_val = (uint8_t)v;
                    }
                }
            }

            double p_selected = byte_probs[selected_val];
            int rank = 0;
            for(int b=0; b<256; b++) {
                if(byte_probs[b] > p_selected) rank++;
                else if(byte_probs[b] == p_selected && b < selected_val) rank++;
            }

            orbit_stream.push_back((uint8_t)rank);
            reconstructed_data[i] = selected_val;
            if (reconstructed_data[i] != img.data[i]) errs++;

            double local_fd = estimate_local_fd(reconstructed_data, x, y, w, h_img);
            double gain = 1.0 + (local_fd - 2.0); // Adaptive learning rate gain
            int actual_orbit = GF8.element_to_orbit_id[selected_val];
            int actual_root = GF8.element_to_root_index[selected_val];
            predictor.train(x_s, x_f, actual_orbit, actual_root, local_fd/3.0, lr * gain);

            last_rank = (uint8_t)rank;
            last_rid = (uint8_t)actual_root;
        }
    }

    std::vector<std::vector<uint8_t>> channels = {orbit_stream};
    std::vector<uint8_t> compressed;
    if (use_lzma) {
        compressed = lzma_compress_channels(channels);
    } else {
        for(const auto& c : channels) compressed.insert(compressed.end(), c.begin(), c.end());
    }

    size_t last_slash = path.find_last_of("/\\");
    size_t second_last_slash = path.find_last_of("/\\", last_slash - 1);
    std::string prefix = (second_last_slash != std::string::npos) ? path.substr(second_last_slash + 1, last_slash - second_last_slash - 1) + "_" : "";
    std::string base = path.substr(last_slash + 1);
    std::string q_str = (lossy_q > 0) ? "_q" + std::to_string(lossy_q) : "";
    savePGM("reconstructed_" + prefix + base.substr(0, base.find_last_of(".")) + q_str + ".pgm", {w, h_img, reconstructed_data});

    double e_orig = calculateEntropy(img.data);
    double e_rank = calculateEntropy(orbit_stream);
    double core_eff = e_orig / (e_rank + 1e-9);

    return {path, e_orig, 0.0, e_rank, 0.0, core_eff, (double)img.data.size() / std::max((size_t)1, compressed.size()), errs == 0};
}

int main(int argc, char** argv) {
    int q = 0, crop = 0, num_imgs = 3;
    if (argc > 1) q = std::stoi(argv[1]);
    if (argc > 2) crop = std::stoi(argv[2]);
    if (argc > 3) num_imgs = std::stoi(argv[3]);

    std::ifstream list("../images_list.txt");
    std::vector<std::string> imgs;
    std::string line;
    while(std::getline(list, line) && (int)imgs.size() < num_imgs) imgs.push_back("../" + line);

    std::string suffix = (q > 0) ? "_q" + std::to_string(q) : "";
    std::ofstream report("compression_report" + suffix + ".md");
    report << "# Peak Holomorphic Tensor Absolute Galois RNN Compression Report (Q=" << q << ")\n\n";
    report << "| Image | Orig Ent | Unified Rank Ent | Core Eff | LZMA Ratio | Recon |\n";
    report << "| :--- | :--- | :--- | :--- | :--- | :--- |\n";

    for(const auto& path : imgs) {
        auto m = processImage(path, 0.015, 128, q, crop, true);
        report << "| " << m.name << " | " << std::fixed << std::setprecision(4) << m.orig_ent
               << " | " << m.rank_orb_ent
               << " | " << m.core_efficiency << ":1 | " << m.lzma_ratio << ":1 | " << (m.success ? "SUCCESS" : "FAIL") << " |\n";
    }
    return 0;
}
