#ifndef MOCK_ARDUINO_HPP
#define MOCK_ARDUINO_HPP

#include <iostream>
#include <vector>
#include <string>
#include <chrono>

class MockSerial {
public:
    void begin(unsigned long baud) {}
    void print(const std::string& s) { std::cout << s; }
    void print(float f) { std::cout << f; }
    void print(int i) { std::cout << i; }
    void println(const std::string& s) { std::cout << s << std::endl; }
    void println(float f) { std::cout << f << std::endl; }
    void println(int i) { std::cout << i << std::endl; }
};

extern MockSerial Serial;

unsigned long millis();
void delay(unsigned long ms);

#endif // MOCK_ARDUINO_HPP
