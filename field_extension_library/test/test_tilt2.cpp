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

FE cameraOffsetX(0.0f);
FE cameraOffsetY(0.0f);
FE cameraTilt(0.0f);
float sceneAngle = 0.0f;
float sceneZoom = 1.0f;

void drawRotazoom() {
    for (int y = 0; y < screenHeight; y += 40) {
        for (int x = 0; x < screenWidth; x += 40) {
            FE fx((x - screenWidth / 2) / float(screenWidth / 2));
            FE fy((y - screenHeight / 2) / float(screenHeight / 2));

            FE theta(sceneAngle);
            FE cosT = cos(theta);
            FE sinT = sin(theta);

            FE rx = cosT * fx;
            rx -= sinT * fy;
            rx /= sceneZoom;

            FE ry = sinT * fx;
            ry += cosT * fy;
            ry /= sceneZoom;

            FE cost = cos(cameraTilt);
            FE sint = sin(cameraTilt);

            FE ux = rx * cost;
            ux -= ry * sint;

            FE uy = rx * sint;
            uy += ry * cost;

            FE vx = ux + cameraOffsetX;
            FE vy = uy + cameraOffsetY;

            int tx = int(vx.toFloat() * 8) & 15;
            int ty = int(vy.toFloat() * 8) & 15;
            tft.drawPixel(x, y, palette[texture[ty][tx]]);
        }
    }
}

int main() {
    std::cout << "Running Refined Tilt 2 Example Test..." << std::endl;
    tft.init();

    float t = 1.0f;
    sceneAngle = 0.5f * t;
    sceneZoom = 1.0f + 0.3f * std::sin(0.7f * t);
    cameraTilt = FE(0.2f * std::sin(1.1f * t));
    cameraOffsetX = FE(0.2f * std::cos(0.9f * t));
    cameraOffsetY = FE(0.2f * std::sin(1.3f * t));

    std::cout << "t=" << t << ", sceneAngle=" << sceneAngle << ", sceneZoom=" << sceneZoom << std::endl;
    drawRotazoom();

    std::cout << "Refined Tilt 2 Example Test Completed Successfully!" << std::endl;
    return 0;
}
