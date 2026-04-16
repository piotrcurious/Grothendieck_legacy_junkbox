#include "Arduino.h"
#include <vector>
#include <iostream>
#include <random>

// Include the .ino file
#include "../exponent_growth_detector.ino"

static std::default_random_engine generator(555);
static std::normal_distribution<double> noise_dist(0.0, 0.2);

int main() {
    setup();

    const int numReadings = 90;
    const float dt = 1.0;

    std::cout << "--- Testing exponent_growth_detector.ino with Saturation (Logistic) Growth ---" << std::endl;
    setMillis(0);
    bufferIndex = 0;
    firstReading = true;

    // Logistic growth: y = L / (1 + exp(-k(t-t0)))
    // L=40, k=0.2, t0=45
    for (int i = 0; i < numReadings; i++) {
        float t = i * dt;
        float temp = 40.0 / (1.0 + exp(-0.2 * (t - 45.0)));
        float noisy_temp = temp + (float)noise_dist(generator);
        float voltage = (noisy_temp / 100.0) + 0.5;
        int raw = (int)(voltage * 1023.0 / 5.0);
        setAnalogReadValue(raw);
        setMillis(i * 1000);
        loop();
    }

    return 0;
}
