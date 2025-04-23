#include <TFT_eSPI.h> #include "FieldExtension.h"

TFT_eSPI tft = TFT_eSPI(); const int screenWidth = 320; const int screenHeight = 240;

// Define pixel colors uint16_t palette[] = { TFT_BLACK, TFT_RED, TFT_ORANGE, TFT_YELLOW, TFT_GREEN, TFT_CYAN, TFT_BLUE, TFT_PURPLE, TFT_WHITE };

// Demo texture: simple 16x16 pattern const uint8_t texture[16][16] = { {0,1,2,3,4,5,6,7,7,6,5,4,3,2,1,0}, {1,2,3,4,5,6,7,0,0,7,6,5,4,3,2,1}, {2,3,4,5,6,7,0,1,1,0,7,6,5,4,3,2}, {3,4,5,6,7,0,1,2,2,1,0,7,6,5,4,3}, {4,5,6,7,0,1,2,3,3,2,1,0,7,6,5,4}, {5,6,7,0,1,2,3,4,4,3,2,1,0,7,6,5}, {6,7,0,1,2,3,4,5,5,4,3,2,1,0,7,6}, {7,0,1,2,3,4,5,6,6,5,4,3,2,1,0,7}, {7,0,1,2,3,4,5,6,6,5,4,3,2,1,0,7}, {6,7,0,1,2,3,4,5,5,4,3,2,1,0,7,6}, {5,6,7,0,1,2,3,4,4,3,2,1,0,7,6,5}, {4,5,6,7,0,1,2,3,3,2,1,0,7,6,5,4}, {3,4,5,6,7,0,1,2,2,1,0,7,6,5,4,3}, {2,3,4,5,6,7,0,1,1,0,7,6,5,4,3,2}, {1,2,3,4,5,6,7,0,0,7,6,5,4,3,2,1}, {0,1,2,3,4,5,6,7,7,6,5,4,3,2,1,0} };

// Camera parameters float camX = 0.0f; float camY = 0.0f; float camTilt = 0.0f; // tilt angle around X-axis

void setup() { tft.init(); tft.setRotation(1); }

void loop() { static float angle = 0.0f; static float scale = 1.0f;

// Update camera translation and tilt over time camX = 50.0f * sin(angle * 0.5f); camY = 30.0f * cos(angle * 0.3f); camTilt = 0.3f * sin(angle * 0.7f);

// Represent transforms using FieldElements FieldElement<4> theta(angle); FieldElement<4> cosTheta = cos(theta); FieldElement<4> sinTheta = sin(theta); FieldElement<4> zoom(scale); FieldElement<4> tilt(camTilt); FieldElement<4> transX(camX); FieldElement<4> transY(camY);

for (int y = 0; y < screenHeight; y++) { for (int x = 0; x < screenWidth; x++) { // Translate to camera and center FieldElement<4> X(x - screenWidth / 2.0f); FieldElement<4> Y(y - screenHeight / 2.0f);

// Apply camera translation
  X = X - transX;
  Y = Y - transY;

  // Apply tilt (simple shear in Y depending on Y depth)
  FieldElement<4> Ytilted = Y + tilt * X;

  // Apply rotazoom on tilted coordinates
  FieldElement<4> rx = (cosTheta * X - sinTheta * Ytilted) / zoom;
  FieldElement<4> ry = (sinTheta * X + cosTheta * Ytilted) / zoom;

  // Map to texture coordinates
  int tx = (int)rx.toFloat() & 15;
  int ty = (int)ry.toFloat() & 15;

  uint8_t colorIndex = texture[ty][tx];
  tft.drawPixel(x, y, palette[colorIndex]);
}

}

// Animate rotazoom angle += 0.02f; scale = 1.0f + 0.5f * sin(angle);

// Frame delay delay(20); }

