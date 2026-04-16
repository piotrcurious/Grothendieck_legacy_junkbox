#include "Arduino.h"
#include <deque>
#include <vector>

#include "../a/2.ino"

int main() {
    setup();
    // Increase the number of loops to allow data collection and fitting
    for(int i = 0; i < 100; i++) {
        // Mock millis needs to advance for loop to trigger sampling
        delay(500);
        loop();
    }
    return 0;
}
