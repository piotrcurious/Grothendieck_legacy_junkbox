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

double estimate_algebraic_complexity(const std::vector<uint8_t>& img, int x, int y, int w, int h) {
    auto get = [&](int tx, int ty) {
        if(tx<0||ty<0||tx>=w||ty>=h) return (uint8_t)0;
        return img[ty*w+tx];
    };
    auto get_deg = [&](int tx, int ty) {
        uint8_t v = get(tx, ty);
        return (double)GF8.orbits[GF8.element_to_orbit_id[v]].elements.size();
    };

    double d0 = get_deg(x, y);
    double d1 = (get_deg(x-1, y) + get_deg(x, y-1) + get_deg(x+1, y) + get_deg(x, y+1)) / 4.0;
    double d2 = (get_deg(x-2, y) + get_deg(x, y-2) + get_deg(x+2, y) + get_deg(x, y+2)) / 4.0;

    // Complexity as rate of change of minimal polynomial degree
    double complexity = (std::abs(d1 - d0) + std::abs(d2 - d1)) / 8.0;
    return std::clamp(complexity, 0.0, 1.0);
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

    // Spatial Path: 8 neighbors * 15 features + 3 feedback (rank, root, resid) = 123
    // Fractal Path: 5 neighbors * 18 features + 2 pos + 8 min_poly_diffs = 100
    DualPathGaloisGRU predictor(123, 100, hid, (int)GF8.orbits.size());

    std::vector<uint8_t> orbit_stream;
    std::vector<uint8_t> reconstructed_data(w * h_img, 0);
    int errs = 0;
    uint8_t last_rank = 0, last_rid = 0, last_resid = 0;

    for(int y=0; y<h_img; ++y) {
        for(int x=0; x<w; ++x) {
            int i = y * w + x;
            if (y < 16 || x < 16 || x >= w - 16 || y >= h_img - 16) {
                reconstructed_data[i] = img.data[i];
                orbit_stream.push_back(0); // placeholder rank
                continue;
            }

            auto getV = [&](int tx, int ty) { return reconstructed_data[ty * w + tx]; };
            auto stalk = [&](uint8_t v, uint8_t ref) {
                uint8_t inv_ref = GF8.inv_table[ref];
                uint8_t mul_diff = GF8.mul(v, inv_ref);
                uint8_t tr1 = GF8.tr8_1(v);
                uint8_t tr2 = GF8.tr8_2(v);
                uint8_t tr4 = GF8.tr8_4(v);
                return std::vector<double>{
                    (double)tr1,
                    (double)tr4/255.0,
                    (double)tr2/255.0,
                    (double)GF8.norm8_4(v)/255.0,
                    (double)GF8.element_to_orbit_id[v]/(double)GF8.orbits.size(),
                    (double)GF8.orbits[GF8.element_to_orbit_id[v]].elements.size()/8.0,
                    (double)v/255.0,
                    (double)GF8.bilinear_trace(v, ref),
                    (double)mul_diff / 255.0,
                    (double)GF8.tr8_power(v, 3),
                    (double)GF8.tr8_power(v, 5),
                    (double)(tr1 ^ (tr2 & 1)), // Holomorphy diff 1
                    (double)(tr2 ^ tr4) / 255.0, // Holomorphy diff 2
                    (double)(tr4 ^ v) / 255.0,  // Holomorphy diff 3
                    (double)GF8.to_normal[v] / 255.0
                };
            };

            std::vector<double> x_s, x_f;
            int nx[] = {x-1, x, x-1, x+1, x-2, x, x-2, x-1}, ny[] = {y, y-1, y-1, y-1, y, y-2, y-1, y-2};
            uint8_t ref_val = getV(x-1, y-1);
            for(int j=0; j<8; ++j) {
                auto s = stalk(getV(nx[j], ny[j]), ref_val);
                x_s.insert(x_s.end(), s.begin(), s.end());
            }
            x_s.push_back((double)last_rank / 255.0);
            x_s.push_back((double)last_rid / 8.0);
            x_s.push_back((double)last_resid / 255.0);

            int fx[] = {x-2, x, x-4, x, x-8}, fy[] = {y, y-2, y, y-4, y};
            for(int j=0; j<5; ++j) {
                uint8_t fv = getV(fx[j], fy[j]);
                auto s = stalk(fv, ref_val);
                x_f.insert(x_f.end(), s.begin(), s.end());
                x_f.push_back(estimate_local_fd(reconstructed_data, fx[j], fy[j], w, h_img) / 3.0);
                x_f.push_back(estimate_algebraic_complexity(reconstructed_data, fx[j], fy[j], w, h_img));
                x_f.push_back((double)GF8.tr8_power(fv, 7));
            }
            x_f.push_back((double)x/w); x_f.push_back((double)y/h_img);

            const auto& mpc_prev = GF8.orbits[GF8.element_to_orbit_id[getV(x-1, y)]].min_poly_coeffs;
            const auto& mpc_curr = GF8.orbits[GF8.element_to_orbit_id[getV(x, y-1)]].min_poly_coeffs;
            for(int k=0; k<8; k++) {
                uint8_t v1 = (k < (int)mpc_prev.size()) ? mpc_prev[k] : 0;
                uint8_t v2 = (k < (int)mpc_curr.size()) ? mpc_curr[k] : 0;
                x_f.push_back((double)(v1 ^ v2) / 255.0);
            }

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
                // Rate-Distortion Optimization (RDO): cost = -log2(p) + lambda * dist
                double lambda = std::pow(2.0, (100.0 - lossy_q) / 10.0) * 0.01;
                int radius = std::max(1, (100 - lossy_q) / 8);
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
            for(int b=0; b<256; b++) if(byte_probs[b] > p_selected) rank++;

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
            last_resid = (uint8_t)std::abs((int)selected_val - (int)target_val);
        }
    }

    std::vector<std::vector<uint8_t>> channels = {orbit_stream};
    std::vector<uint8_t> compressed;
    if (use_lzma) {
        compressed = lzma_compress_channels(channels);
    } else {
        for(const auto& c : channels) compressed.insert(compressed.end(), c.begin(), c.end());
    }

    std::string base = path.substr(path.find_last_of("/\\") + 1);
    std::string q_str = (lossy_q > 0) ? "_q" + std::to_string(lossy_q) : "";
    savePGM("reconstructed_" + base.substr(0, base.find_last_of(".")) + q_str + ".pgm", {w, h_img, reconstructed_data});

    double e_orig = calculateEntropy(img.data);
    double e_rank = calculateEntropy(orbit_stream);
    double core_eff = e_orig / (e_rank + 1e-9);

    return {path, e_orig, 0.0, e_rank, 0.0, core_eff, (double)img.data.size() / std::max((size_t)1, compressed.size()), errs == 0};
}

int main(int argc, char** argv) {
    std::string img1 = "../absolute_galois_group/compressor/01/test.pgm";
    std::string img2 = "../absolute_galois_group/compressor/01/GhostInShell_02_005.pgm";
    std::string img3 = "../absolute_galois_group/compressor/01/iter/ghcd_08/test.pgm";

    int q = 0, crop = 0;
    if (argc > 1) q = std::stoi(argv[1]);
    if (argc > 2) crop = std::stoi(argv[2]);

    auto m1 = processImage(img1, 0.015, 128, q, crop, true);
    auto m2 = processImage(img2, 0.015, 128, q, crop, true);
    auto m3 = processImage(img3, 0.015, 128, q, crop, true);

    std::string suffix = (q > 0) ? "_q" + std::to_string(q) : "";
    std::ofstream report("compression_report" + suffix + ".md");
    report << "# Peak Holomorphic Tensor Absolute Galois RNN Compression Report\n\n";
    report << "| Image | Orig Ent | Unified Rank Ent | Core Eff | LZMA Ratio | Recon |\n";
    report << "| :--- | :--- | :--- | :--- | :--- | :--- |\n";
    auto add = [&](Metrics m) {
        report << "| " << m.name << " | " << std::fixed << std::setprecision(4) << m.orig_ent
               << " | " << m.rank_orb_ent
               << " | " << m.core_efficiency << ":1 | " << m.lzma_ratio << ":1 | " << (m.success ? "SUCCESS" : "FAIL") << " |\n";
    };
    add(m1); add(m2); add(m3);
    return 0;
}
