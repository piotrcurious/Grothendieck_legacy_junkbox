#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <algorithm>
#include "rnn_absolute_core.h"

struct Image {
    int width, height;
    std::vector<uint8_t> data;
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

void compress(const std::string& in, const std::string& out, int lossy_q) {
    Image img = loadPGM(in);
    int w = img.width, h_img = img.height;
    // Spatial (146), Fractal (100)
    DualPathGaloisGRU predictor(146, 100, 128, (int)GF8.orbits.size());
    std::vector<uint8_t> rank_stream;
    std::vector<uint8_t> recon(w * h_img, 0);
    uint8_t last_rank = 0, last_rid = 0;
    SignalContext ctx = {w, h_img, recon};

    for(int y=0; y<h_img; ++y) {
        for(int x=0; x<w; ++x) {
            int i = y * w + x;
            if (y < 16 || x < 16 || x >= w - 16 || y >= h_img - 16) {
                recon[i] = img.data[i];
                rank_stream.push_back(img.data[i]);
                continue;
            }

            std::vector<double> x_s, x_f;
            ctx.get_features(x, y, last_rank, last_rid, x_s, x_f);
            auto res_pair = predictor.forward(x_s, x_f);

            std::vector<double> byte_probs(256, 0.0);
            std::vector<double> orbit_root_sums(GF8.orbits.size(), 1e-9);
            for(int b=0; b<256; b++) {
                int oid = GF8.element_to_orbit_id[b];
                int rid = GF8.element_to_root_index[b];
                orbit_root_sums[oid] += (rid < 8) ? res_pair.p_root[rid] : 0.0;
            }
            for(int b=0; b<256; b++) {
                int oid = GF8.element_to_orbit_id[b], rid = GF8.element_to_root_index[b];
                byte_probs[b] = res_pair.p_orb[oid] * ((rid < 8) ? (res_pair.p_root[rid] / orbit_root_sums[oid]) : 0.0);
            }

            uint8_t target_val = img.data[i], selected_val = target_val;
            if (lossy_q > 0) {
                double q_norm = (double)lossy_q / 100.0;
                double lambda = std::pow(10.0, 5.0 * q_norm - 1.0);
                int radius = std::max(1, (int)(24.0 * (1.0 - q_norm)));
                double min_cost = 1e18;
                for (int v = std::max(0, (int)target_val - radius); v <= std::min(255, (int)target_val + radius); ++v) {
                    double cost = -std::log2(std::max(byte_probs[v], 1e-12)) + lambda * (v - target_val) * (v - target_val);
                    if (cost < min_cost) { min_cost = cost; selected_val = (uint8_t)v; }
                }
            }

            double p_selected = byte_probs[selected_val]; int rank = 0;
            for(int b=0; b<256; b++) {
                if(byte_probs[b] > p_selected) rank++;
                else if(byte_probs[b] == p_selected && b < selected_val) rank++;
            }

            rank_stream.push_back((uint8_t)rank);
            recon[i] = selected_val;
            double lfd = estimate_local_fd(recon, x, y, w, h_img);
            predictor.train(x_s, x_f, GF8.element_to_orbit_id[selected_val], GF8.element_to_root_index[selected_val], lfd/3.0, 0.015 * (1.0 + lfd - 2.0));

            last_rank = (uint8_t)rank;
            last_rid = (uint8_t)GF8.element_to_root_index[selected_val];
        }
    }

    std::vector<uint8_t> header = {(uint8_t)(w & 0xFF), (uint8_t)((w >> 8) & 0xFF), (uint8_t)(h_img & 0xFF), (uint8_t)((h_img >> 8) & 0xFF)};
    auto compressed = lzma_compress_channels({header, rank_stream});
    std::ofstream f(out, std::ios::binary); f.write((char*)compressed.data(), compressed.size());
}

void decompress(const std::string& in, const std::string& out) {
    std::ifstream f(in, std::ios::binary | std::ios::ate);
    std::streamsize size = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<uint8_t> comp(size); f.read((char*)comp.data(), size);
    auto channels = lzma_decompress_channels(comp);
    if (channels.size() < 2) return;
    int w = channels[0][0] | (channels[0][1] << 8), h_img = channels[0][2] | (channels[0][3] << 8);
    const auto& rank_stream = channels[1];

    DualPathGaloisGRU predictor(178, 100, 128, (int)GF8.orbits.size());
    Image img; img.width = w; img.height = h_img; img.data.assign(w * h_img, 0);
    uint8_t last_rank = 0, last_rid = 0;
    int stream_pos = 0;
    SignalContext ctx = {w, h_img, img.data};

    for(int y=0; y<h_img; ++y) {
        for(int x=0; x<w; ++x) {
            int i = y * w + x;
            if (y < 16 || x < 16 || x >= w - 16 || y >= h_img - 16) {
                img.data[i] = rank_stream[stream_pos++];
                continue;
            }

            std::vector<double> x_s, x_f;
            ctx.get_features(x, y, last_rank, last_rid, x_s, x_f);
            auto res_pair = predictor.forward(x_s, x_f);

            std::vector<double> orbit_root_sums(GF8.orbits.size(), 1e-9);
            for(int b=0; b<256; b++) {
                int oid = GF8.element_to_orbit_id[b];
                int rid = GF8.element_to_root_index[b];
                orbit_root_sums[oid] += (rid < 8) ? res_pair.p_root[rid] : 0.0;
            }

            std::vector<std::pair<double, uint8_t>> probs(256);
            for(int b=0; b<256; b++) {
                int oid = GF8.element_to_orbit_id[b], rid = GF8.element_to_root_index[b];
                double p = res_pair.p_orb[oid] * ((rid < 8) ? (res_pair.p_root[rid] / orbit_root_sums[oid]) : 0.0);
                probs[b] = {p, (uint8_t)b};
            }

            std::sort(probs.begin(), probs.end(), [](const auto& a, const auto& b) {
                if(a.first != b.first) return a.first > b.first;
                return a.second < b.second;
            });

            uint8_t rank = rank_stream[stream_pos++];
            uint8_t val = probs[rank].second;
            img.data[i] = val;

            double lfd = estimate_local_fd(img.data, x, y, w, h_img);
            predictor.train(x_s, x_f, GF8.element_to_orbit_id[val], GF8.element_to_root_index[val], lfd/3.0, 0.015 * (1.0 + lfd - 2.0));

            last_rank = rank;
            last_rid = (uint8_t)GF8.element_to_root_index[val];
        }
    }
    savePGM(out, img);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " compress <in.pgm> <out.rnn> [quality]\n";
        std::cout << "       " << argv[0] << " decompress <in.rnn> <out.pgm>\n";
        return 1;
    }
    std::string mode = argv[1];
    try {
        if (mode == "compress") compress(argv[2], argv[3], (argc > 4 ? std::stoi(argv[4]) : 0));
        else if (mode == "decompress") decompress(argv[2], argv[3]);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
