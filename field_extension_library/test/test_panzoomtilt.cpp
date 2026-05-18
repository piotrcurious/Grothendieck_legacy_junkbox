#include "Arduino.h"
#include "TFT_eSPI.h"
#include "../FieldExtension.h"
#include <iostream>

using FE = FieldElement<16>;
TFT_eSPI tft = TFT_eSPI();
const int screenWidth = 320;
const int screenHeight = 240;

uint16_t palette[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
const uint8_t texture[16][16] = { {0} };

float camX = 0.0f;
float camY = 0.0f;
float camTilt = 0.0f;

int main() {
    std::cout << "Running Refined Panzoomtilt Example Test..." << std::endl;
    tft.init();

    float angle = 0.0f;
    float scale = 1.0f;

    for(int frame = 0; frame < 2; frame++) {
        camX = 50.0f * std::sin(angle * 0.5f);
        camY = 30.0f * std::cos(angle * 0.3f);
        camTilt = 0.3f * std::sin(angle * 0.7f);

        FE theta(angle);
        FE cosTheta = cos(theta);
        FE sinTheta = sin(theta);
        FE zoom(scale);
        FE tilt(camTilt);
        FE transX(camX);
        FE transY(camY);

        std::cout << "Frame " << frame << ": camX=" << camX << ", camY=" << camY << ", tilt=" << camTilt << std::endl;

        for (int y = 0; y < screenHeight; y += 40) {
            for (int x = 0; x < screenWidth; x += 40) {
                FE X(x - screenWidth / 2.0f);
                FE Y(y - screenHeight / 2.0f);

                X -= transX;
                Y -= transY;

                FE Ytilted = Y + tilt * X;

                FE rx = cosTheta * X;
                rx -= sinTheta * Ytilted;
                rx /= zoom;

                FE ry = sinTheta * X;
                ry += cosTheta * Ytilted;
                ry /= zoom;

                int tx = (int)rx.toFloat() & 15;
                int ty = (int)ry.toFloat() & 15;

                tft.drawPixel(x, y, palette[texture[ty][tx]]);
            }
        }
        angle += 0.02f;
        scale = 1.0f + 0.5f * std::sin(angle);
    }

    std::cout << "Refined Panzoomtilt Example Test Completed Successfully!" << std::endl;
    return 0;
}
