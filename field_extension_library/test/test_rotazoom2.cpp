#include "Arduino.h"
#include "TFT_eSPI.h"
#include "../FieldExtension.h"
#include <iostream>

using FE = FieldElement16;

TFT_eSPI tft = TFT_eSPI();
const int screenWidth = 320;
const int screenHeight = 240;

uint16_t palette[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
const uint8_t texture[16][16] = { {0} };

int main() {
    std::cout << "Running Refined Rotazoom 2 Test..." << std::endl;
    tft.init();

    float angle = 0.0f;
    float scale = 1.0f;

    for(int frame = 0; frame < 2; frame++) {
        FE theta(angle);
        FE cosTheta = cos(theta);
        FE sinTheta = sin(theta);
        FE zoom(scale);

        std::cout << "Frame " << frame << ": angle=" << angle << ", scale=" << scale << std::endl;

        for (int y = 0; y < screenHeight; y += 40) {
            for (int x = 0; x < screenWidth; x += 40) {
                float fx = x - screenWidth / 2;
                float fy = y - screenHeight / 2;

                FE X(fx);
                FE Y(fy);

                FE rx = cosTheta * X;
                rx -= sinTheta * Y;
                rx /= zoom;

                FE ry = sinTheta * X;
                ry += cosTheta * Y;
                ry /= zoom;

                int tx = int(rx.toFloat()) & 15;
                int ty = int(ry.toFloat()) & 15;

                tft.drawPixel(x, y, palette[texture[ty][tx]]);
            }
        }
        angle += 0.02f;
        scale = 1.0f + 0.5f * std::sin(angle);
    }

    std::cout << "Refined Rotazoom 2 Test Completed Successfully!" << std::endl;
    return 0;
}
