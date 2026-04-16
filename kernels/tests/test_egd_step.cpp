#include "Arduino.h"
#include <vector>
#include <iostream>
#include <random>

// Include the .ino file
#include "../exponent_growth_detector.ino"

static std::default_random_engine generator(999);
static std::normal_distribution<double> noise_dist(0.0, 0.2);

int main() {
    setup();

    const int numReadings = 90; // 3 buffers
    const float dt = 1.0;

    std::cout << "--- Testing exponent_growth_detector.ino with Step Change Data ---" << std::endl;
    setMillis(0);
    bufferIndex = 0;
    firstReading = true;
    for (int i = 0; i < numReadings; i++) {
        float temp = (i < 45) ? 20.0 : 40.0; // Sudden step at 45 seconds
        float noisy_temp = temp + (float)noise_dist(generator);
        float voltage = (noisy_temp / 100.0) + 0.5;
        int raw = (int)(voltage * 1023.0 / 5.0);
        setAnalogReadValue(raw);

        setMillis(i * 1000);
        loop();
    }

    return 0;
}
