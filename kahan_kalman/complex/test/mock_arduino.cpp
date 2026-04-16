#include "Arduino.h"
#include <chrono>
#include <thread>
#include <random>
#include <vector>
MockSerial Serial;
static unsigned long mock_millis = 0;
static int next_analog_value = 512;
static std::vector<int> analog_sequence;
static size_t sequence_index = 0;
void delay(unsigned long ms) { mock_millis += ms; }
unsigned long millis() { return mock_millis; }
void resetMockTime() { mock_millis = 0; }
void setMockAnalogRead(int value) { next_analog_value = value; analog_sequence.clear(); }
void setMockAnalogSequence(const std::vector<int>& sequence) { analog_sequence = sequence; sequence_index = 0; }
int analogRead(uint8_t pin) {
    if (sequence_index < analog_sequence.size()) return analog_sequence[sequence_index++];
    return next_analog_value;
}
static std::default_random_engine generator;
long random(long min, long max) {
    if (min >= max) return min;
    std::uniform_int_distribution<long> distribution(min, max - 1);
    return distribution(generator);
}
void randomSeed(unsigned long seed) { generator.seed(seed); }
void analogReadResolution(int res) {}
void analogSetAttenuation(int atten) {}
