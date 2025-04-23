#include <TFT_eSPI.h> #include "FieldExtension.h"

TFT_eSPI tft = TFT_eSPI(); const int screenWidth = 320; const int screenHeight = 240;

// Define pixel colors uint16_t palette[] = { TFT_BLACK, TFT_RED, TFT_ORANGE, TFT_YELLOW, TFT_GREEN, TFT_CYAN, TFT_BLUE, TFT_PURPLE, TFT_WHITE };

// Demo texture: simple 16x16 pattern const uint8_t texture[16][16] = { {0,1,2,3,4,5,6,7,7,6,5,4,3,2,1,0}, {1,2,3,4,5,6,7,0,0,7,6,5,4,3,2,1}, {2,3,4,5,6,7,0,1,1,0,7,6,5,4,3,2}, {3,4,5,6,7,0,1,2,2,1,0,7,6,5,4,3}, {4,5,6,7,0,1,2,3,3,2,1,0,7,6,5,4}, {5,6,7,0,1,2,3,4,4,3,2,1,0,7,6,5}, {6,7,0,1,2,3,4,5,5,4,3,2,1,0,7,6}, {7,0,1,2,3,4,5,6,6,5,4,3,2,1,0,7}, {7,0,1,2,3,4,5,6,6,5,4,3,2,1,0,7}, {6,7,0,1,2,3,4,5,5,4,3,2,1,0,7,6}, {5,6,7,0,1,2,3,4,4,3,2,1,0,7,6,5}, {4,5,6,7,0,1,2,3,3,2,1,0,7,6,5,4}, {3,4,5,6,7,0,1,2,2,1,0,7,6,5,4,3}, {2,3,4,5,6,7,0,1,1,0,7,6,5,4,3,2}, {1,2,3,4,5,6,7,0,0,7,6,5,4,3,2,1}, {0,1,2,3,4,5,6,7,7,6,5,4,3,2,1,0} };

// Camera translation offsets float camX = 0.0f; float camY = 0.0f;

void setup() { tft.init(); tft.setRotation(1); }

void loop() { static float angle = 0.0f; static float scale = 1.0f;

// Represent transformations using field elements FieldElement<4> theta(angle); FieldElement<4> cosTheta = cos(theta); FieldElement<4> sinTheta = sin(theta); FieldElement<4> zoom(scale); FieldElement<4> transX(camX); FieldElement<4> transY(camY);

// Animate camera in a circular path camX = 50.0f * cos(angle * 0.5f); camY = 30.0f * sin(angle * 0.5f);

// Update field elements after animation transX.setCoefficient(0, camX); transY.setCoefficient(0, camY);

// Loop through screen pixels for (int y = 0; y < screenHeight; y++) { for (int x = 0; x < screenWidth; x++) { // Translate pixel to center FieldElement<4> X(float(x - screenWidth/2)); FieldElement<4> Y(float(y - screenHeight/2));

// Apply camera translation
  X = X - transX;
  Y = Y - transY;

  // Apply rotation and zoom: X' = (cosθ·X - sinθ·Y) / zoom
  FieldElement<4> rx = (cosTheta * X - sinTheta * Y) / zoom;
  FieldElement<4> ry = (sinTheta * X + cosTheta * Y) / zoom;

  // Convert to texture coordinates
  int tx = int(rx.toFloat()) & 15;
  int ty = int(ry.toFloat()) & 15;

  uint8_t colorIndex = texture[ty][tx];
  tft.drawPixel(x, y, palette[colorIndex]);
}

}

// Update angle and scale angle += 0.02f; scale = 1.0f + 0.5f * sin(angle);

delay(20); }

