#ifndef INTERFERENCE_CORE_H
#define INTERFERENCE_CORE_H

#include <vector>
#include <thread>
#include <random>
#include <complex>
#include <algorithm>
#include "complex_math.h"

namespace interference {

    /**
     * compute_threaded: A generic, multi-threaded root density generator.
     * Supports custom mapping and weighting.
     */
    template<typename Mapper, typename Weighter>
    void compute_weighted_threaded(int total_samples, int res, int max_deg, int max_c,
                                  std::vector<double>& heat, Mapper map_func, Weighter weight_func,
                                  bool use_max = false, double cx = 0.0, double cy = 0.0, double range = 2.0) {
        int num_threads = std::thread::hardware_concurrency();
        if (num_threads < 1) num_threads = 1;
        int samples_per_thread = total_samples / num_threads;

        std::vector<std::vector<double>> thread_heats(num_threads, std::vector<double>(res * res, use_max ? -1e9 : 0.0));
        std::vector<std::thread> threads;

        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back([&, t]() {
                std::mt19937 trng(1337 + t);
                std::uniform_int_distribution<int> tc_dist(-max_c, max_c);
                int start = t * samples_per_thread;
                int end = (t == num_threads - 1) ? total_samples : (t + 1) * samples_per_thread;

                for (int i = start; i < end; ++i) {
                    int d = (i % max_deg) + 1;
                    std::vector<cd> coeffs(d + 1);
                    for (int k = 0; k <= d; ++k) {
                        int c = tc_dist(trng);
                        if (k == d && c == 0) c = 1;
                        coeffs[k] = cd((double)c, 0.0);
                    }
                    auto roots = dk_solve_roots(coeffs);
                    for (const auto& r : roots) {
                        cd mr = map_func(r);
                        double w = weight_func(r);
                        int ix = (int)((mr.real() - (cx - range)) / (2.0 * range) * (res - 1));
                        int iy = (int)((mr.imag() - (cy - range)) / (2.0 * range) * (res - 1));
                        if (ix >= 0 && ix < res && iy >= 0 && iy < res) {
                            if (use_max) {
                                if (w > thread_heats[t][iy * res + ix]) thread_heats[t][iy * res + ix] = w;
                            } else {
                                thread_heats[t][iy * res + ix] += w;
                            }
                        }
                    }
                }
            });
        }
        for (auto& th : threads) th.join();

        if (use_max) {
            std::fill(heat.begin(), heat.end(), -1e9);
            for (int t = 0; t < num_threads; ++t) {
                for (int i = 0; i < res * res; ++i) {
                    if (thread_heats[t][i] > heat[i]) heat[i] = thread_heats[t][i];
                }
            }
            // Normalize -1e9 back to 0 or something reasonable for display
            for (double &v : heat) if (v < -1e8) v = 0;
        } else {
            std::fill(heat.begin(), heat.end(), 0.0);
            for (int t = 0; t < num_threads; ++t) {
                for (int i = 0; i < res * res; ++i) heat[i] += thread_heats[t][i];
            }
        }
    }

    // Convenience for backward compatibility
    template<typename Mapper>
    void compute_threaded(int total_samples, int res, int max_deg, int max_c, std::vector<double>& heat, Mapper map_func, double cx = 0.0, double cy = 0.0, double range = 2.0) {
        compute_weighted_threaded(total_samples, res, max_deg, max_c, heat, map_func, [](cd){ return 1.0; }, false, cx, cy, range);
    }

    inline void magma_color(double t, unsigned char* rgb) {
        t = (t < 0) ? 0 : (t > 1 ? 1 : t);
        rgb[0] = (unsigned char)(pow(t, 0.4) * 255);
        rgb[1] = (unsigned char)(pow(t, 1.8) * 180);
        rgb[2] = (unsigned char)(pow(1.0 - t, 1.2) * 220);
    }

    inline void plasma_color(double t, unsigned char* rgb) {
        t = (t < 0) ? 0 : (t > 1 ? 1 : t);
        rgb[0] = (unsigned char)(pow(t, 0.3) * 255);
        rgb[1] = (unsigned char)(t * 150);
        rgb[2] = (unsigned char)((1.0 - pow(t, 2.0)) * 255);
    }

    inline void viridis_color(double t, unsigned char* rgb) {
        t = (t < 0) ? 0 : (t > 1 ? 1 : t);
        rgb[0] = (unsigned char)((1.0 - t) * 70);
        rgb[1] = (unsigned char)(t * 220);
        rgb[2] = (unsigned char)(pow(t, 0.5) * 150 + 50);
    }

    inline void apply_color(int type, double t, unsigned char* rgb) {
        switch(type) {
            case 1: plasma_color(t, rgb); break;
            case 2: viridis_color(t, rgb); break;
            case 0:
            default: magma_color(t, rgb); break;
        }
    }
}

#endif
