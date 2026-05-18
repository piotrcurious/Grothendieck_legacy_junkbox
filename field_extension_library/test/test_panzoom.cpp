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

int main() {
    std::cout << "Running Refined Panzoom Example Test..." << std::endl;
    tft.init();

    float angle = 0.0f;
    float scale = 1.0f;

    for(int frame = 0; frame < 2; frame++) {
        FE theta(angle);
        FE cosTheta = cos(theta);
        FE sinTheta = sin(theta);
        FE zoom(scale);
        FE transX(camX);
        FE transY(camY);

        camX = 50.0f * std::cos(angle * 0.5f);
        camY = 30.0f * std::sin(angle * 0.5f);

        transX.setCoefficient(0, camX);
        transY.setCoefficient(0, camY);

        std::cout << "Frame " << frame << ": camX=" << camX << ", camY=" << camY << std::endl;

        for (int y = 0; y < screenHeight; y += 40) {
            for (int x = 0; x < screenWidth; x += 40) {
                FE X(float(x - screenWidth/2));
                FE Y(float(y - screenHeight/2));

                X -= transX;
                Y -= transY;

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

    std::cout << "Refined Panzoom Example Test Completed Successfully!" << std::endl;
    return 0;
}
