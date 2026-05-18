#include "Arduino.h"
#include "TFT_eSPI.h"
#include "../FieldExtension.h"
#include <iostream>

TFT_eSPI tft = TFT_eSPI();
const int screenWidth = 320;
const int screenHeight = 240;

uint16_t palette[] = { TFT_BLACK, TFT_RED, TFT_ORANGE, TFT_YELLOW, TFT_GREEN, TFT_CYAN, TFT_BLUE, TFT_PURPLE, TFT_WHITE };

const uint8_t texture[16][16] = { {0,1,2,3,4,5,6,7,7,6,5,4,3,2,1,0}, {1,2,3,4,5,6,7,0,0,7,6,5,4,3,2,1}, {2,3,4,5,6,7,0,1,1,0,7,6,5,4,3,2}, {3,4,5,6,7,0,1,2,2,1,0,7,6,5,4,3}, {4,5,6,7,0,1,2,3,3,2,1,0,7,6,5,4}, {5,6,7,0,1,2,3,4,4,3,2,1,0,7,6,5}, {6,7,0,1,2,3,4,5,5,4,3,2,1,0,7,6}, {7,0,1,2,3,4,5,6,6,5,4,3,2,1,0,7}, {7,0,1,2,3,4,5,6,6,5,4,3,2,1,0,7}, {6,7,0,1,2,3,4,5,5,4,3,2,1,0,7,6}, {5,6,7,0,1,2,3,4,4,3,2,1,0,7,6,5}, {4,5,6,7,0,1,2,3,3,2,1,0,7,6,5,4}, {3,4,5,6,7,0,1,2,2,1,0,7,6,5,4,3}, {2,3,4,5,6,7,0,1,1,0,7,6,5,4,3,2}, {1,2,3,4,5,6,7,0,0,7,6,5,4,3,2,1}, {0,1,2,3,4,5,6,7,7,6,5,4,3,2,1,0} };

FieldElement<4> cameraOffsetX(0.0f);
FieldElement<4> cameraOffsetY(0.0f);
FieldElement<4> cameraTilt(0.0f);
float sceneAngle = 0.0f;
float sceneZoom = 1.0f;

void drawRotazoom() {
    for (int y = 0; y < screenHeight; y += 40) {
        for (int x = 0; x < screenWidth; x += 40) {
            FieldElement<4> fx((x - screenWidth / 2) / float(screenWidth / 2));
            FieldElement<4> fy((y - screenHeight / 2) / float(screenHeight / 2));

            FieldElement<4> theta(sceneAngle);
            FieldElement<4> cosT = cos(theta);
            FieldElement<4> sinT = sin(theta);
            FieldElement<4> rx = (cosT * fx - sinT * fy) / FieldElement<4>(sceneZoom);
            FieldElement<4> ry = (sinT * fx + cosT * fy) / FieldElement<4>(sceneZoom);

            FieldElement<4> cost = cos(cameraTilt);
            FieldElement<4> sint = sin(cameraTilt);
            FieldElement<4> ux = rx * cost - ry * sint;
            FieldElement<4> uy = rx * sint + ry * cost;

            FieldElement<4> vx = ux + cameraOffsetX;
            FieldElement<4> vy = uy + cameraOffsetY;

            int tx = int(vx.toFloat() * 8) & 15;
            int ty = int(vy.toFloat() * 8) & 15;
            uint8_t ci = texture[ty][tx];
            tft.drawPixel(x, y, palette[ci]);
        }
    }
}

int main() {
    std::cout << "Running Tilt 2 Example Test..." << std::endl;
    tft.init();

    float t = 1.0f;
    sceneAngle = 0.5f * t;
    sceneZoom = 1.0f + 0.3f * std::sin(0.7f * t);
    cameraTilt = FieldElement<4>(0.2f * std::sin(1.1f * t));
    cameraOffsetX = FieldElement<4>(0.2f * std::cos(0.9f * t));
    cameraOffsetY = FieldElement<4>(0.2f * std::sin(1.3f * t));

    std::cout << "t=" << t << ", sceneAngle=" << sceneAngle << ", sceneZoom=" << sceneZoom << std::endl;
    drawRotazoom();

    std::cout << "Tilt 2 Example Test Completed Successfully!" << std::endl;
    return 0;
}
