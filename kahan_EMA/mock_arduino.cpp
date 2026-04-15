#include "Arduino.h"
#include <chrono>
#include <thread>
#include <random>
#include <cstdarg>

MockSerial Serial;

void delay(unsigned long ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

unsigned long millis() {
    static auto start = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
}

int analogRead(uint8_t pin) {
    return 512; // Mock value
}

long random(long min, long max) {
    static std::default_random_engine generator;
    std::uniform_int_distribution<long> distribution(min, max - 1);
    return distribution(generator);
}
