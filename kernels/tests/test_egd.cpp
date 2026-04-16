#include "Arduino.h"
#include <vector>
#include <iostream>

// Include the .ino file - we'll need to wrap it or modify it slightly if it has name collisions,
// but for now let's try direct inclusion.
// Note: .ino files are basically C++, but we might need to declare functions if they are used before definition.
// In our case, the .ino files are well-structured.

#include "../exponent_growth_detector.ino"

int main() {
    setup();

    // Simulation parameters
    const int numReadings = 60; // 2 full buffers
    const float dt = 1.0; // 1 second interval

    std::cout << "--- Testing exponent_growth_detector.ino with Linear Data ---" << std::endl;
    setMillis(0);
    for (int i = 0; i < numReadings; i++) {
        // Linear data: T = 20 + 0.5 * t
        float t = i * dt;
        float temp = 20.0 + 0.5 * t;
        // Convert temp back to raw analog value
        // temperatureC = (voltage - 0.5) * 100.0;
        // voltage = rawValue * (5.0 / 1023.0);
        float voltage = (temp / 100.0) + 0.5;
        int raw = (int)(voltage * 1023.0 / 5.0);
        setAnalogReadValue(raw);

        setMillis(i * 1000);
        loop();
    }

    std::cout << "\n--- Testing exponent_growth_detector.ino with Exponential Data ---" << std::endl;
    setMillis(0);
    bufferIndex = 0; // Reset buffer index for clean test
    for (int i = 0; i < numReadings; i++) {
        // Exponential data: T = 20 * exp(0.05 * t)
        float t = i * dt;
        float temp = 20.0 * exp(0.05 * t);
        float voltage = (temp / 100.0) + 0.5;
        int raw = (int)(voltage * 1023.0 / 5.0);
        setAnalogReadValue(raw);

        setMillis(i * 1000);
        loop();
    }

    return 0;
}
