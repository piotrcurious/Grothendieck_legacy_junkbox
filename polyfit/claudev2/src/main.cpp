#include "mock_arduino.hpp"
#include "arduino_polyfit.hpp"

// Prototypes for Arduino-like entry points
void setup();
void loop();

int main() {
    setup();
    // Run loop a few times for the mock
    for (int i = 0; i < 1; ++i) {
        loop();
    }
    return 0;
}
