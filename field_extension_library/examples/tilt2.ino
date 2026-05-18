#include <TFT_eSPI.h> #include "FieldExtension.h"

// TFT display instance 
TFT_eSPI tft = TFT_eSPI(); const int screenWidth = 320; const int screenHeight = 240;

// Color palette 
uint16_t palette[] = { TFT_BLACK, TFT_RED, TFT_ORANGE, TFT_YELLOW, TFT_GREEN, TFT_CYAN, TFT_BLUE, TFT_PURPLE, TFT_WHITE };

// 16x16 demo texture pattern 
const uint8_t texture[16][16] = { {0,1,2,3,4,5,6,7,7,6,5,4,3,2,1,0}, {1,2,3,4,5,6,7,0,0,7,6,5,4,3,2,1}, {2,3,4,5,6,7,0,1,1,0,7,6,5,4,3,2}, {3,4,5,6,7,0,1,2,2,1,0,7,6,5,4,3}, {4,5,6,7,0,1,2,3,3,2,1,0,7,6,5,4}, {5,6,7,0,1,2,3,4,4,3,2,1,0,7,6,5}, {6,7,0,1,2,3,4,5,5,4,3,2,1,0,7,6}, {7,0,1,2,3,4,5,6,6,5,4,3,2,1,0,7}, {7,0,1,2,3,4,5,6,6,5,4,3,2,1,0,7}, {6,7,0,1,2,3,4,5,5,4,3,2,1,0,7,6}, {5,6,7,0,1,2,3,4,4,3,2,1,0,7,6,5}, {4,5,6,7,0,1,2,3,3,2,1,0,7,6,5,4}, {3,4,5,6,7,0,1,2,2,1,0,7,6,5,4,3}, {2,3,4,5,6,7,0,1,1,0,7,6,5,4,3,2}, {1,2,3,4,5,6,7,0,0,7,6,5,4,3,2,1}, {0,1,2,3,4,5,6,7,7,6,5,4,3,2,1,0} };

// Camera and scene parameters 
FieldElement<16> cameraOffsetX(0.0f);
FieldElement<16> cameraOffsetY(0.0f);
FieldElement<16> cameraTilt(0.0f);  // tilt angle in radians
float sceneAngle = 0.0f;
float sceneZoom = 1.0f;

// Draw the rotazoomer with camera translation and tilt 
void drawRotazoom() {
    tft.startWrite();
    for (int y = 0; y < screenHeight; y++) {
        for (int x = 0; x < screenWidth; x++) {
            // Center and normalize coordinates
            FieldElement<16> fx((x - screenWidth / 2) / float(screenWidth / 2));
            FieldElement<16> fy((y - screenHeight / 2) / float(screenHeight / 2));

            // Scene rotation & zoom
            FieldElement<16> theta(sceneAngle);
            FieldElement<16> cosT = cos(theta);
            FieldElement<16> sinT = sin(theta);

            FieldElement<16> rx = cosT * fx;
            rx -= sinT * fy;
            rx /= sceneZoom;

            FieldElement<16> ry = sinT * fx;
            ry += cosT * fy;
            ry /= sceneZoom;

            // Camera tilt
            FieldElement<16> cost = cos(cameraTilt);
            FieldElement<16> sint = sin(cameraTilt);

            FieldElement<16> ux = rx * cost;
            ux -= ry * sint;

            FieldElement<16> uy = rx * sint;
            uy += ry * cost;

            // Camera translation (panning)
            FieldElement<16> vx = ux + cameraOffsetX;
            FieldElement<16> vy = uy + cameraOffsetY;

  // Texture lookup
  int tx = int(vx.toFloat() * 8) & 15;
  int ty = int(vy.toFloat() * 8) & 15;
  uint8_t ci = texture[ty][tx];
  tft.drawPixel(x, y, palette[ci]);
}

} tft.endWrite(); }

void setup() { tft.init(); tft.setRotation(1); }

void loop() { float t = millis() * 0.001f; // Animate scene sceneAngle = 0.5f * t; sceneZoom = 1.0f + 0.3f * sin(0.7f * t); // Animate camera cameraTilt = FieldElement<4>(0.2f * sin(1.1f * t)); cameraOffsetX = FieldElement<4>(0.2f * cos(0.9f * t)); cameraOffsetY = FieldElement<4>(0.2f * sin(1.3f * t));

drawRotazoom(); delay(20); }

