#include <TFT_eSPI.h>
#include <math.h>

TFT_eSPI tft = TFT_eSPI();

#define SCREEN_WIDTH  240
#define SCREEN_HEIGHT 320

// Parameters for Lissajous curve and animation
float a = 3.0, b = 2.0, delta = PI / 2.0;
float perturbationAmplitude = 0.3;
float time = 0.0;
int maxResolution = 12;  // Maximum tessellation resolution
int minResolution = 4;   // Minimum tessellation resolution

// Function to compute a Lissajous point with perturbations
void computeLissajousPoint(float t, float &x, float &y) {
  // Base Lissajous equation
  x = sin(a * t + delta);
  y = cos(b * t);

  // Add perturbations
  x += perturbationAmplitude * sin(5 * t + time);
  y += perturbationAmplitude * cos(4 * t - time);
}

// Map coordinates to screen
void mapToScreen(float x, float y, float &screenX, float &screenY) {
  screenX = map(x, -1.5, 1.5, 20, SCREEN_WIDTH - 20);
  screenY = map(y, -1.5, 1.5, 20, SCREEN_HEIGHT - 20);
}

// Draw a triangle
void drawTriangle(float x1, float y1, float x2, float y2, float x3, float y3, uint16_t color) {
  tft.drawLine(x1, y1, x2, y2, color);
  tft.drawLine(x2, y2, x3, y3, color);
  tft.drawLine(x3, y3, x1, y1, color);
}

// Draw tessellation shapes (triangles and hexagons)
void tessellateSurface(int resolution, uint16_t color) {
  float step = TWO_PI / resolution;

  for (int i = 0; i < resolution; i++) {
    for (int j = 0; j < resolution; j++) {
      // Grid positions (parametric t values)
      float t1 = i * step;
      float t2 = (i + 1) * step;
      float t3 = j * step;
      float t4 = (j + 1) * step;

      // Points on the perturbed Lissajous curve
      float x1, y1, x2, y2, x3, y3, x4, y4;
      computeLissajousPoint(t1, x1, y1);
      computeLissajousPoint(t2, x2, y2);
      computeLissajousPoint(t3, x3, y3);
      computeLissajousPoint(t4, x4, y4);

      // Map points to screen
      mapToScreen(x1, y1, x1, y1);
      mapToScreen(x2, y2, x2, y2);
      mapToScreen(x3, y3, x3, y3);
      mapToScreen(x4, y4, x4, y4);

      // Alternate between triangle and hexagon shapes
      if ((i + j) % 2 == 0) {
        drawTriangle(x1, y1, x2, y2, x3, y3, color);
      } else {
        float cx = (x1 + x2 + x3 + x4) / 4;
        float cy = (y1 + y2 + y3 + y4) / 4;
        float radius = min(abs(x2 - x1), abs(y3 - y1)) / 2;
        tft.fillCircle(cx, cy, radius, color);
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

// Main loop
void loop() {
  tft.fillScreen(TFT_BLACK); // Clear the screen

  // Update animation parameters
  time += 0.05;
  perturbationAmplitude = 0.2 + 0.15 * sin(time);

  // Progressively tessellate with varying resolutions
  for (int resolution = minResolution; resolution <= maxResolution; resolution++) {
    uint16_t color = tft.color565((resolution * 20) % 255, (resolution * 40) % 255, (resolution * 60) % 255);
    tessellateSurface(resolution, color);
  }

  delay(50); // Control animation speed
}
