#include "Arduino.h"
#include <vector>
#include <iostream>
#include <random>

// Include the .ino file
#include "../EGD_2pass.ino"

static std::default_random_engine generator(123);
static std::normal_distribution<double> noise_dist(0.0, 0.5);

float add_noise(float temp) {
    return temp + (float)noise_dist(generator);
}

int main() {
    setup();

    const int numReadings = 60;
    const float dt = 1.0;

    std::cout << "--- Testing EGD_2pass.ino with Noisy Exponential Data ---" << std::endl;
    setMillis(0);
    bufferIndex = 0;
    for (int i = 0; i < numReadings; i++) {
        float t = i * dt;
        float temp = 20.0 * exp(0.05 * t);
        float noisy_temp = add_noise(temp);
        float voltage = (noisy_temp / 100.0) + 0.5;
        int raw = (int)(voltage * 1023.0 / 5.0);
        setAnalogReadValue(raw);

        setMillis(i * 1000);
        loop();
    }

    std::cout << "\n--- Testing EGD_2pass.ino with Noisy Linear Data ---" << std::endl;
    setMillis(0);
    bufferIndex = 0;
    for (int i = 0; i < numReadings; i++) {
        float t = i * dt;
        float temp = 20.0 + 0.5 * t;
        float noisy_temp = add_noise(temp);
        float voltage = (noisy_temp / 100.0) + 0.5;
        int raw = (int)(voltage * 1023.0 / 5.0);
        setAnalogReadValue(raw);

        setMillis(i * 1000);
        loop();
    }

    return 0;
}
