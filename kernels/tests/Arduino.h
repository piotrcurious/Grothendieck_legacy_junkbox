#ifndef ARDUINO_H
#define ARDUINO_H
#include <iostream>
#include <string>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <cstdarg>
#include <vector>

#define A0 0

class MockSerial {
public:
    void begin(unsigned long baud) {}
    void print(const std::string& s) { std::cout << s; }
    void print(const char* s) { std::cout << s; }
    void print(float f) { std::cout << f; }
    void print(double d) { std::cout << d; }
    void print(int i) { std::cout << i; }
    void print(unsigned long l) { std::cout << l; }
    void print(double d, int p) { std::cout << std::fixed << std::setprecision(p) << d; }
    void println() { std::cout << std::endl; }
    void println(const std::string& s) { std::cout << s << std::endl; }
    void println(const char* s) { std::cout << s << std::endl; }
    void println(float f) { std::cout << f << std::endl; }
    void println(double d) { std::cout << d << std::endl; }
    void println(int i) { std::cout << i << std::endl; }
    void println(unsigned long l) { std::cout << l << std::endl; }
    void println(double d, int p) { std::cout << std::fixed << std::setprecision(p) << d << std::endl; }
    void printf(const char* format, ...) {
        va_list args;
        va_start(args, format);
        vprintf(format, args);
        va_end(args);
    }
    operator bool() const { return true; }
};

extern MockSerial Serial;

void delay(unsigned long ms);
unsigned long millis();
int analogRead(uint8_t pin);

// Helper for tests
void setAnalogReadValue(int value);
void setMillis(unsigned long ms);

#endif
