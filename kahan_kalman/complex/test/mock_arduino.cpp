#include "Arduino.h"
#include <chrono>
#include <thread>
#include <random>
MockSerial Serial;
void delay(unsigned long ms) {}
unsigned long global_mock_millis = 10000;
unsigned long millis() {
    global_mock_millis += 1000;
    return global_mock_millis;
}
int analogRead(uint8_t pin) {
    return 512;
}
static std::default_random_engine generator;
long random(long min, long max) {
    if (min >= max) return min;
    std::uniform_int_distribution<long> distribution(min, max - 1);
    return distribution(generator);
}
void randomSeed(unsigned long seed) {
    generator.seed(seed);
}
void analogReadResolution(int res) {}
void analogSetAttenuation(int atten) {}
