#include "Arduino.h"
#include "TFT_eSPI.h"
#include "../FieldExtension.h"
#include <iostream>

using FE = FieldElement16;

TFT_eSPI tft = TFT_eSPI();
static const int16_t W = 320;
static const int16_t H = 240;
static const float CX = W / 2.0f;
static const float CY = H / 2.0f;

FE baseAngle() {
  FE a;
  a.setCoefficient(1, 1.0f);   // π
  a.setCoefficient(2, 0.5f);   // e
  a.setCoefficient(3, 0.2f);   // √2
  return a;
}

int main() {
    std::cout << "Running Refined Rotazoom Test..." << std::endl;
    tft.init();

    float t = 0.0f;
    for(int frame = 0; frame < 2; frame++) {
        t += 0.02f;
        FE ang = baseAngle();
        ang.setCoefficient(0, t);

        FE s = sin(ang);
        FE c = cos(ang);

        FE zoomFE = 1.0f + 0.5f * sin(FE(t * 0.5f));
        float zoom = zoomFE.toFloat();

        float fs = s.toFloat();
        float fc = c.toFloat();

        std::cout << "Frame " << frame << ": t=" << t << ", zoom=" << zoom << ", sin=" << fs << ", cos=" << fc << std::endl;

        for (int16_t y = 0; y < H; y += 40) {
            for (int16_t x = 0; x < W; x += 40) {
                float u = (x - CX) * zoom;
                float v = (y - CY) * zoom;
                float xr = u * fc - v * fs;
                float yr = u * fs + v * fc;
                tft.drawPixel(x, y, 0);
            }
        }
    }

    std::cout << "Refined Rotazoom Test Completed Successfully!" << std::endl;
    return 0;
}
