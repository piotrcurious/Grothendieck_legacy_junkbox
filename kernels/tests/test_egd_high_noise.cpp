#include "Arduino.h"
#include <vector>
#include <iostream>
#include <random>

// Include the .ino file
#include "../exponent_growth_detector.ino"

static std::default_random_engine generator(777);
static std::normal_distribution<double> high_noise(0.0, 2.0); // SD = 2.0 degrees

int main() {
    setup();

    const int numReadings = 60;
    const float dt = 1.0;

    std::cout << "--- Testing exponent_growth_detector.ino with High Noise Linear Data ---" << std::endl;
    setMillis(0);
    bufferIndex = 0;
    firstReading = true;
    for (int i = 0; i < numReadings; i++) {
        float t = i * dt;
        float temp = 20.0 + 0.5 * t;
        float noisy_temp = temp + (float)high_noise(generator);
        float voltage = (noisy_temp / 100.0) + 0.5;
        int raw = (int)(voltage * 1023.0 / 5.0);
        setAnalogReadValue(raw);
        setMillis(i * 1000);
        loop();
    }

    std::cout << "\n--- Testing exponent_growth_detector.ino with High Noise Exponential Data ---" << std::endl;
    setMillis(0);
    bufferIndex = 0;
    firstReading = true;
    for (int i = 0; i < numReadings; i++) {
        float t = i * dt;
        float temp = 20.0 * exp(0.05 * t);
        float noisy_temp = temp + (float)high_noise(generator);
        float voltage = (noisy_temp / 100.0) + 0.5;
        int raw = (int)(voltage * 1023.0 / 5.0);
        setAnalogReadValue(raw);
        setMillis(i * 1000);
        loop();
    }

    return 0;
}
