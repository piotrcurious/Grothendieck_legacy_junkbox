#include "Arduino.h"
#include <vector>
#include <iostream>

// Include the .ino file
#include "../EGD_2pass.ino"

int main() {
    setup();

    const int numReadings = 60;
    const float dt = 1.0;

    std::cout << "--- Testing EGD_2pass.ino with Linear Data ---" << std::endl;
    setMillis(0);
    for (int i = 0; i < numReadings; i++) {
        float t = i * dt;
        float temp = 20.0 + 0.5 * t;
        float voltage = (temp / 100.0) + 0.5;
        int raw = (int)(voltage * 1023.0 / 5.0);
        setAnalogReadValue(raw);

        setMillis(i * 1000);
        loop();
    }

    std::cout << "\n--- Testing EGD_2pass.ino with Exponential Data ---" << std::endl;
    setMillis(0);
    bufferIndex = 0;
    for (int i = 0; i < numReadings; i++) {
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
