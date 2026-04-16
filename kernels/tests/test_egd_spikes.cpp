#include "Arduino.h"
#include <vector>
#include <iostream>
#include <random>

// Include the .ino file
#include "../exponent_growth_detector.ino"

static std::default_random_engine generator(42);
static std::normal_distribution<double> noise_dist(0.0, 0.5);

float add_noise_and_spikes(float temp, int i) {
    float val = temp + (float)noise_dist(generator);
    if (i % 15 == 7) { // Every 15th reading is a spike
        val += 20.0;
    }
    return val;
}

int main() {
    setup();

    const int numReadings = 60;
    const float dt = 1.0;

    std::cout << "--- Testing exponent_growth_detector.ino with Spiky Linear Data ---" << std::endl;
    setMillis(0);
    bufferIndex = 0;
    firstReading = true; // Reset EMA
    for (int i = 0; i < numReadings; i++) {
        float t = i * dt;
        float temp = 20.0 + 1.0 * t; // Linear growth
        float noisy_temp = add_noise_and_spikes(temp, i);
        float voltage = (noisy_temp / 100.0) + 0.5;
        int raw = (int)(voltage * 1023.0 / 5.0);
        setAnalogReadValue(raw);

        setMillis(i * 1000);
        loop();
    }

    std::cout << "\n--- Testing exponent_growth_detector.ino with Spiky Exponential Data ---" << std::endl;
    setMillis(0);
    bufferIndex = 0;
    firstReading = true; // Reset EMA
    for (int i = 0; i < numReadings; i++) {
        float t = i * dt;
        float temp = 20.0 * exp(0.05 * t);
        float noisy_temp = add_noise_and_spikes(temp, i);
        float voltage = (noisy_temp / 100.0) + 0.5;
        int raw = (int)(voltage * 1023.0 / 5.0);
        setAnalogReadValue(raw);

        setMillis(i * 1000);
        loop();
    }

    return 0;
}
