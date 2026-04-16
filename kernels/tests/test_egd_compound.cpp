#include "Arduino.h"
#include <vector>
#include <iostream>
#include <random>

// Include the .ino file
#include "../exponent_growth_detector.ino"

static std::default_random_engine generator(111);
static std::normal_distribution<double> noise_dist(0.0, 0.3);

int main() {
    setup();

    const int numReadings = 120;
    const float dt = 1.0;

    std::cout << "--- Testing exponent_growth_detector.ino with Compound Growth (Linear then Exponential) ---" << std::endl;
    setMillis(0);
    bufferIndex = 0;
    firstReading = true;

    // Simulation: Linear for first 60 (2 full buffers), then Exponential for 60
    for (int i = 0; i < numReadings; i++) {
        float t = i * dt;
        float temp;
        if (i < 60) {
            temp = 20.0 + 1.0 * t;
        } else {
            temp = 80.0 * exp(0.05 * (t-60));
        }
        float noisy_temp = temp + (float)noise_dist(generator);
        float voltage = (noisy_temp / 100.0) + 0.5;
        int raw = (int)(voltage * 1023.0 / 5.0);
        setAnalogReadValue(raw);
        setMillis(i * 1000);
        loop();
    }

    return 0;
}
