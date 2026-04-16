#ifndef ARDUINO_H
#define ARDUINO_H
#include <iostream>
#include <string>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <cstdarg>
#include <cstring>
#include <vector>
#include <algorithm>

#define GPIO_NUM_34 34
#define F(s) s

class MockSerial {
public:
    void begin(unsigned long baud) {}
    void print(const char* s) { std::cout << s; }
    void print(const std::string& s) { std::cout << s; }
    void print(float f) { std::cout << f; }
    void print(double d) { std::cout << d; }
    void print(int i) { std::cout << i; }
    void print(unsigned int i) { std::cout << i; }
    void print(long i) { std::cout << i; }
    void print(unsigned long i) { std::cout << i; }

    void println() { std::cout << std::endl; }
    void println(const char* s) { std::cout << s << std::endl; }
    void println(const std::string& s) { std::cout << s << std::endl; }
    void println(float f) { std::cout << f << std::endl; }
    void println(double d) { std::cout << d << std::endl; }
    void println(int i) { std::cout << i << std::endl; }
    void println(unsigned int i) { std::cout << i << std::endl; }
    void println(long i) { std::cout << i << std::endl; }
    void println(unsigned long i) { std::cout << i << std::endl; }

    void printf(const char* format, ...) {
        va_list args; va_start(args, format);
        vprintf(format, args); va_end(args);
    }
    operator bool() const { return true; }
};

extern MockSerial Serial;

void delay(unsigned long ms);
unsigned long millis();
int analogRead(uint8_t pin);
long random(long min, long max);
void randomSeed(unsigned long seed);
void resetMockTime();

using std::min;
using std::max;
using std::isnan;
using std::isinf;
using std::abs;

#endif
