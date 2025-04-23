#include <TFT_eSPI.h>
#include "FieldExtension.h"

TFT_eSPI tft = TFT_eSPI();
const int screenWidth = 320;
const int screenHeight = 240;

// Define pixel colors
uint16_t palette[] = {
  TFT_BLACK, TFT_RED, TFT_ORANGE, TFT_YELLOW,
  TFT_GREEN, TFT_CYAN, TFT_BLUE, TFT_PURPLE, TFT_WHITE
};

// Demo texture: simple 16x16 pattern
const uint8_t texture[16][16] = {
  {0,1,2,3,4,5,6,7,7,6,5,4,3,2,1,0},
  {1,2,3,4,5,6,7,0,0,7,6,5,4,3,2,1},
  {2,3,4,5,6,7,0,1,1,0,7,6,5,4,3,2},
  {3,4,5,6,7,0,1,2,2,1,0,7,6,5,4,3},
  {4,5,6,7,0,1,2,3,3,2,1,0,7,6,5,4},
  {5,6,7,0,1,2,3,4,4,3,2,1,0,7,6,5},
  {6,7,0,1,2,3,4,5,5,4,3,2,1,0,7,6},
  {7,0,1,2,3,4,5,6,6,5,4,3,2,1,0,7},
  {7,0,1,2,3,4,5,6,6,5,4,3,2,1,0,7},
  {6,7,0,1,2,3,4,5,5,4,3,2,1,0,7,6},
  {5,6,7,0,1,2,3,4,4,3,2,1,0,7,6,5},
  {4,5,6,7,0,1,2,3,3,2,1,0,7,6,5,4},
  {3,4,5,6,7,0,1,2,2,1,0,7,6,5,4,3},
  {2,3,4,5,6,7,0,1,1,0,7,6,5,4,3,2},
  {1,2,3,4,5,6,7,0,0,7,6,5,4,3,2,1},
  {0,1,2,3,4,5,6,7,7,6,5,4,3,2,1,0}
};

void setup() {
  tft.init();
  tft.setRotation(1);
}

void loop() {
  static float angle = 0.0f;
  static float scale = 1.0f;

  // Represent angle and scale using field elements
  FieldElement<4> theta(angle);
  FieldElement<4> cosTheta = cos(theta);
  FieldElement<4> sinTheta = sin(theta);
  FieldElement<4> zoom(scale);

  // Loop through screen pixels
  for (int y = 0; y < screenHeight; y++) {
    for (int x = 0; x < screenWidth; x++) {
      // Translate to center
      float fx = x - screenWidth / 2;
      float fy = y - screenHeight / 2;

      // Convert to FieldElements
      FieldElement<4> X(fx);
      FieldElement<4> Y(fy);

      // Apply rotazoom: X' = cosθ·X - sinθ·Y, Y' = sinθ·X + cosθ·Y
      FieldElement<4> rx = (cosTheta * X - sinTheta * Y) / zoom;
      FieldElement<4> ry = (sinTheta * X + cosTheta * Y) / zoom;

      // Map to texture coordinates
      int tx = int(rx.toFloat()) & 15;
      int ty = int(ry.toFloat()) & 15;

      uint8_t colorIndex = texture[ty][tx];
      tft.drawPixel(x, y, palette[colorIndex]);
    }
  }

  // Update angle and scale
  angle += 0.02;
  scale = 1.0f + 0.5f * sin(angle);

  // Optional: reduce flickering by syncing frame updates
  delay(20);
}
