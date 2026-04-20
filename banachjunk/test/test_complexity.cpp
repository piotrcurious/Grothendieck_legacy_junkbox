#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "Arduino.h"
#include "test_data_generator.h"
#include "../math_utils.h"

void test_sampen_robustness() {
    std::cout << "Testing SampEn and Sparsity metrics..." << std::endl;

    // Case 1: Pure Sine (Low complexity, Low sparsity)
    std::vector<float> sine_v;
    for(int i=0; i<100; ++i) sine_v.push_back(std::sin(i*0.1f));

    float se_sine = banach::Statistics::calculateSampEn(sine_v);
    float sp_sine = banach::Statistics::calculateSparsity(sine_v);
    std::cout << "Sine: SampEn=" << se_sine << ", Sparsity=" << sp_sine << std::endl;
    assert(se_sine < 0.5f);
    assert(sp_sine < 0.3f);

    // Case 2: White Noise (High complexity, Mid sparsity)
    std::vector<float> noise_v;
    for(int i=0; i<100; ++i) noise_v.push_back((random(2000)-1000)/1000.0f);
    float se_noise = banach::Statistics::calculateSampEn(noise_v);
    float sp_noise = banach::Statistics::calculateSparsity(noise_v);
    std::cout << "Noise: SampEn=" << se_noise << ", Sparsity=" << sp_noise << std::endl;
    assert(se_noise > se_sine);

    // Case 3: Sparse Spike (Low complexity, High sparsity)
    std::vector<float> spike_v(100, 0.01f);
    spike_v[50] = 10.0f;
    float se_spike = banach::Statistics::calculateSampEn(spike_v);
    float sp_spike = banach::Statistics::calculateSparsity(spike_v);
    std::cout << "Spike: SampEn=" << se_spike << ", Sparsity=" << sp_spike << std::endl;
    assert(sp_spike > 0.8f);
}

int main() {
    test_sampen_robustness();
    std::cout << "Complexity tests passed!" << std::endl;
    return 0;
}
