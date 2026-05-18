#include "Arduino.h"
#include "TFT_eSPI.h"
#include "../FieldExtension.h"
#include <iostream>

using FE = FieldElement<4>;
TFT_eSPI tft = TFT_eSPI();
const int screenWidth = 320;
const int screenHeight = 240;

uint16_t palette[] = { TFT_BLACK, TFT_RED, TFT_ORANGE, TFT_YELLOW, TFT_GREEN, TFT_CYAN, TFT_BLUE, TFT_PURPLE, TFT_WHITE };

const uint8_t texture[16][16] = { {0,1,2,3,4,5,6,7,7,6,5,4,3,2,1,0}, {1,2,3,4,5,6,7,0,0,7,6,5,4,3,2,1}, {2,3,4,5,6,7,0,1,1,0,7,6,5,4,3,2}, {3,4,5,6,7,0,1,2,2,1,0,7,6,5,4,3}, {4,5,6,7,0,1,2,3,3,2,1,0,7,6,5,4}, {5,6,7,0,1,2,3,4,4,3,2,1,0,7,6,5}, {6,7,0,1,2,3,4,5,5,4,3,2,1,0,7,6}, {7,0,1,2,3,4,5,6,6,5,4,3,2,1,0,7}, {7,0,1,2,3,4,5,6,6,5,4,3,2,1,0,7}, {6,7,0,1,2,3,4,5,5,4,3,2,1,0,7,6}, {5,6,7,0,1,2,3,4,4,3,2,1,0,7,6,5}, {4,5,6,7,0,1,2,3,3,2,1,0,7,6,5,4}, {3,4,5,6,7,0,1,2,2,1,0,7,6,5,4,3}, {2,3,4,5,6,7,0,1,1,0,7,6,5,4,3,2}, {1,2,3,4,5,6,7,0,0,7,6,5,4,3,2,1}, {0,1,2,3,4,5,6,7,7,6,5,4,3,2,1,0} };

float camX = 0.0f;
float camY = 0.0f;
float camTilt = 0.0f;

int main() {
    std::cout << "Running Panzoomtilt Example Test..." << std::endl;
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

                X = X - transX;
                Y = Y - transY;

                FE Ytilted = Y + tilt * X;

                FE rx = (cosTheta * X - sinTheta * Ytilted) / zoom;
                FE ry = (sinTheta * X + cosTheta * Ytilted) / zoom;

                int tx = (int)rx.toFloat() & 15;
                int ty = (int)ry.toFloat() & 15;

                uint8_t colorIndex = texture[ty][tx];
                tft.drawPixel(x, y, palette[colorIndex]);
            }
        }
        angle += 0.02f;
        scale = 1.0f + 0.5f * std::sin(angle);
    }

    std::cout << "Panzoomtilt Example Test Completed Successfully!" << std::endl;
    return 0;
}
