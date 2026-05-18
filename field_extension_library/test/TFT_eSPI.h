#pragma once
#include "Arduino.h"

#define TFT_BLACK 0x0000
#define TFT_RED   0xF800
#define TFT_ORANGE 0xFD20
#define TFT_YELLOW 0xFFE0
#define TFT_GREEN 0x07E0
#define TFT_CYAN  0x07FF
#define TFT_BLUE  0x001F
#define TFT_PURPLE 0x780F
#define TFT_WHITE 0xFFFF

class TFT_eSPI {
public:
    TFT_eSPI(int w = 320, int h = 240) {}
    void init() {}
    void setRotation(int r) {}
    void fillScreen(uint16_t color) {}
    void setCursor(int x, int y) {}
    void setTextColor(uint16_t color) {}
    void setTextSize(int size) {}
    void print(const char* s) { std::cout << s; }
    void print(const String& s) { std::cout << s; }
    void printf(const char* format, ...) {
        va_list args;
        va_start(args, format);
        vprintf(format, args);
        va_end(args);
    }
    void drawRect(int x, int y, int w, int h, uint16_t color) {}
    void drawPixel(int x, int y, uint16_t color) {}
    void fillCircle(int x, int y, int r, uint16_t color) {}
    uint16_t color565(uint8_t r, uint8_t g, uint8_t b) { return 0; }
};
