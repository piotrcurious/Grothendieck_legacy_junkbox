#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cstdio>
#include <stdarg.h>
#include <cstdint>
#include <array>
#include <random>

// Use inline functions instead of macros to avoid conflicts with std
#undef max
#undef min

template<typename T, typename U>
auto max(T a, U b) -> decltype(a > b ? a : b) {
    return (a > b) ? a : b;
}

template<typename T, typename U>
auto min(T a, U b) -> decltype(a < b ? a : b) {
    return (a < b) ? a : b;
}

class String : public std::string {
public:
    String() : std::string() {}
    String(const char* s) : std::string(s) {}
    String(const std::string& s) : std::string(s) {}

    float toFloat() const {
        try {
            return std::stof(*this);
        } catch (...) {
            return 0.0f;
        }
    }

    int toInt() const {
        try {
            return std::stoi(*this);
        } catch (...) {
            return 0;
        }
    }

    const char* c_str() const {
        return std::string::c_str();
    }
};

class SerialMock {
public:
    void begin(unsigned long baud) {}
    void print(const char* s) { std::cout << s; }
    void print(const String& s) { std::cout << s; }
    void print(float f) { std::cout << f; }
    void print(double d) { std::cout << d; }
    void print(int i) { std::cout << i; }
    void println(const char* s) { std::cout << s << std::endl; }
    void println(const String& s) { std::cout << s << std::endl; }
    void println(float f) { std::cout << f << std::endl; }
    void println(double d) { std::cout << d << std::endl; }
    void println(int i) { std::cout << i << std::endl; }
    void println() { std::cout << std::endl; }
    void printf(const char* format, ...) {
        va_list args;
        va_start(args, format);
        vprintf(format, args);
        va_end(args);
    }
};

extern SerialMock Serial;

void delay(unsigned long ms);
unsigned long millis();
unsigned long micros();

#define F(s) s
#define PI 3.14159265358979323846

inline long random(long max) {
    static std::mt19937 gen(0);
    std::uniform_int_distribution<long> dis(0, max - 1);
    return dis(gen);
}

inline long random(long min, long max) {
    static std::mt19937 gen(0);
    std::uniform_int_distribution<long> dis(min, max - 1);
    return dis(gen);
}
