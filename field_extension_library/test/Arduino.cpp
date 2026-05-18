#include "Arduino.h"
#include <chrono>
#include <thread>

SerialMock Serial;

static auto start_time = std::chrono::steady_clock::now();
static unsigned long mock_time_ms = 0;
static bool use_real_time = false;

void delay(unsigned long ms) {
    if (use_real_time) {
        std::this_thread::sleep_for(std::chrono::milliseconds(ms));
    } else {
        mock_time_ms += ms;
    }
}

unsigned long millis() {
    if (use_real_time) {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
    }
    return mock_time_ms;
}

unsigned long micros() {
    if (use_real_time) {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(now - start_time).count();
    }
    return mock_time_ms * 1000;
}
