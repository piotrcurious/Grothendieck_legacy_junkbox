#pragma once
#include "Arduino.h"

struct GPSDate {
    bool isValid() { return true; }
    int year() { return 2023; }
    int month() { return 10; }
    int day() { return 27; }
    struct { int day = 27; } value;
};

struct GPSTime {
    bool isValid() { return true; }
    int hour() { return 12; }
    int minute() { return 0; }
    int second() { return 0; }
};

struct GPSLocation {
    bool isValid() { return true; }
    double lat() { return 37.7749; }
    double lng() { return -122.4194; }
};

class TinyGPSPlus {
public:
    static GPSDate date;
    static GPSTime time;
    static GPSLocation location;
    void encode(char c) {}
};

inline GPSDate TinyGPSPlus::date;
inline GPSTime TinyGPSPlus::time;
inline GPSLocation TinyGPSPlus::location;

class HardwareSerial {
public:
    HardwareSerial(int port) {}
    void begin(unsigned long baud, uint32_t config = 0, int rx = -1, int tx = -1) {}
    bool available() { return false; }
    char read() { return 0; }
};

#define SERIAL_8N1 0
