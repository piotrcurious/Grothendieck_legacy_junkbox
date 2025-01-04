#include <TFT_eSPI.h>  // Include TFT_eSPI library
#include <math.h>      // For sin, cos functions

TFT_eSPI tft = TFT_eSPI(); // Create TFT instance

// Screen dimensions
#define SCREEN_WIDTH  240
#define SCREEN_HEIGHT 320

// Parameters for animation
float a = 3.0, b = 2.0;     // Lissajous curve coefficients
int maxResolution = 10;     // Maximum tessellation resolution
float perturbationAmplitude = 0.3; // Amplitude of perturbations
float time = 0.0;           // Animation time step

// Function to compute a perturbed Lissajous surface point
void computePerturbedLissajous(float u, float v, float &x, float &y) {
  // Base Lissajous curve
  x = sin(a * u + time) + perturbationAmplitude * sin(5 * u + time) +
      perturbationAmplitude * 0.5 * cos(3 * v - time);
  y = cos(b * v - time) + perturbationAmplitude * sin(4 * v - time) +
      perturbationAmplitude * 0.5 * cos(6 * u + time);
}

// Map coordinates to the screen with proper scaling
void mapToScreen(float x, float y, float &screenX, float &screenY) {
  screenX = map(x, -2.0, 2.0, 0, SCREEN_WIDTH);
  screenY = map(y, -2.0, 2.0, 0, SCREEN_HEIGHT);
}

// Draw a triangle
void drawTriangle(float x1, float y1, float x2, float y2, float x3, float y3, uint16_t color) {
  tft.drawLine(x1, y1, x2, y2, color);
  tft.drawLine(x2, y2, x3, y3, color);
  tft.drawLine(x3, y3, x1, y1, color);
}

// Draw a rectangle
void drawRectangle(float x1, float y1, float x2, float y2, float x3, float y3, float x4, float y4, uint16_t color) {
  tft.drawLine(x1, y1, x2, y2, color);
  tft.drawLine(x2, y2, x3, y3, color);
  tft.drawLine(x3, y3, x4, y4, color);
  tft.drawLine(x4, y4, x1, y1, color);
}

// Draw a hexagon
void drawHexagon(float cx, float cy, float radius, uint16_t color) {
  for (int i = 0; i < 6; i++) {
    float angle1 = i * 60 * DEG_TO_RAD;
    float angle2 = (i + 1) * 60 * DEG_TO_RAD;
    float x1 = cx + radius * cos(angle1);
    float y1 = cy + radius * sin(angle1);
    float x2 = cx + radius * cos(angle2);
    float y2 = cy + radius * sin(angle2);
    tft.drawLine(x1, y1, x2, y2, color);
  }
}

// Tessellate the surface dynamically
void tessellateSurface(int resolution, uint16_t color) {
  float du = TWO_PI / resolution;
  float dv = TWO_PI / resolution;

  for (int i = 0; i < resolution; i++) {
    for (int j = 0; j < resolution; j++) {
      // Compute (u, v) for the current grid cell
      float u1 = i * du, v1 = j * dv;
      float u2 = (i + 1) * du, v2 = v1;
      float u3 = u1, v3 = (j + 1) * dv;
      float u4 = u2, v4 = v3;

      // Compute screen coordinates for the vertices
      float x1, y1, x2, y2, x3, y3, x4, y4;
      computePerturbedLissajous(u1, v1, x1, y1);
      computePerturbedLissajous(u2, v2, x2, y2);
      computePerturbedLissajous(u3, v3, x3, y3);
      computePerturbedLissajous(u4, v4, x4, y4);

      mapToScreen(x1, y1, x1, y1);
      mapToScreen(x2, y2, x2, y2);
      mapToScreen(x3, y3, x3, y3);
      mapToScreen(x4, y4, x4, y4);

      // Alternate shapes dynamically
      if ((i + j + (int)time) % 3 == 0) {
        drawTriangle(x1, y1, x2, y2, x3, y3, color);
      } else if ((i + j + (int)time) % 3 == 1) {
        drawRectangle(x1, y1, x2, y2, x4, y4, x3, y3, color);
      } else {
        float cx = (x1 + x2 + x3 + x4) / 4;
        float cy = (y1 + y2 + y3 + y4) / 4;
        float radius = min(abs(x2 - x1), abs(y3 - y1)) / 2;
        drawHexagon(cx, cy, radius, color);
      }
    }
  }
}

// Setup
void setup() {
  tft.init();
  tft.setRotation(1);
  tft.fillScreen(TFT_BLACK);
}

// Loop for animation
void loop() {
  tft.fillScreen(TFT_BLACK); // Clear screen

  // Update animation parameters
  time += 0.05;
  perturbationAmplitude = 0.3 + 0.1 * sin(time); // Vary perturbation amplitude over time

  // Illustrate cofibration by layering tessellations
  for (int resolution = 4; resolution <= maxResolution; resolution += 2) {
    uint16_t color = (resolution % 2 == 0) ? TFT_GREEN : TFT_BLUE;
    tessellateSurface(resolution, color);
  }

  delay(50); // Animation speed
}
