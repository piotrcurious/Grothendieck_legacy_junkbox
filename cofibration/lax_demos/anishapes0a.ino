#include <TFT_eSPI.h>
#include <math.h>

TFT_eSPI tft = TFT_eSPI();  // Create TFT object

#define SCREEN_WIDTH  240
#define SCREEN_HEIGHT 320

// Animation parameters
float time = 0.0;
float perturbationAmplitude = 0.3;  // Amplitude of perturbation for surface
float radius1 = 1.0;  // Major radius for torus
float radius2 = 0.4;  // Minor radius for torus

// Function to compute the perturbed torus surface (base surface)
void computeTorusSurface(float u, float v, float &x, float &y, float &z) {
  // Parametric equations for a torus
  x = (radius1 + radius2 * cos(v)) * cos(u);
  y = (radius1 + radius2 * cos(v)) * sin(u);
  z = radius2 * sin(v);

  // Perturbations applied to surface
  x += perturbationAmplitude * sin(4 * u + time);
  y += perturbationAmplitude * cos(3 * v + time);
  z += perturbationAmplitude * sin(2 * u + v + time);
}

// Map 3D coordinates to 2D screen coordinates
void mapToScreen(float x, float y, float &screenX, float &screenY) {
  // Simple perspective projection: map from 3D to 2D (x, y)
  screenX = map(x, -1.5, 1.5, 20, SCREEN_WIDTH - 20);
  screenY = map(y, -1.5, 1.5, 20, SCREEN_HEIGHT - 20);
}

// Function to draw a triangle (used in tessellation)
void drawTriangle(float x1, float y1, float x2, float y2, float x3, float y3, uint16_t color) {
  tft.drawLine(x1, y1, x2, y2, color);
  tft.drawLine(x2, y2, x3, y3, color);
  tft.drawLine(x3, y3, x1, y1, color);
}

// Function to tessellate the surface with triangles
void tessellateSurface(int resolution, uint16_t color) {
  float du = 2 * M_PI / resolution;  // Step size in u-direction
  float dv = 2 * M_PI / resolution;  // Step size in v-direction

  for (int i = 0; i < resolution; i++) {
    for (int j = 0; j < resolution; j++) {
      // Parametric u and v values for the current tessellation grid
      float u1 = i * du, v1 = j * dv;
      float u2 = (i + 1) * du, v2 = v1;
      float u3 = u1, v3 = (j + 1) * dv;
      float u4 = u2, v4 = v3;

      // Compute the perturbed torus surface points
      float x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4;
      computeTorusSurface(u1, v1, x1, y1, z1);
      computeTorusSurface(u2, v2, x2, y2, z2);
      computeTorusSurface(u3, v3, x3, y3, z3);
      computeTorusSurface(u4, v4, x4, y4, z4);

      // Map the surface points to screen space (2D projection)
      float screenX1, screenY1, screenX2, screenY2, screenX3, screenY3, screenX4, screenY4;
      mapToScreen(x1, y1, screenX1, screenY1);
      mapToScreen(x2, y2, screenX2, screenY2);
      mapToScreen(x3, y3, screenX3, screenY3);
      mapToScreen(x4, y4, screenX4, screenY4);

      // Draw tessellations (triangles)
      if ((i + j) % 2 == 0) {
        drawTriangle(screenX1, screenY1, screenX2, screenY2, screenX3, screenY3, color);
      } else {
        drawTriangle(screenX2, screenY2, screenX3, screenY3, screenX4, screenY4, color);
      }
    }
  }
}

// Setup function
void setup() {
  tft.init();
  tft.setRotation(1);
  tft.fillScreen(TFT_BLACK);
}

// Main loop
void loop() {
  tft.fillScreen(TFT_BLACK);  // Clear the screen

  // Update animation parameters
  time += 0.05;
  perturbationAmplitude = 0.3 + 0.15 * sin(time);  // Varying perturbation amplitude for dynamism

  // Tessellate surface with varying resolution
  for (int resolution = 4; resolution <= 12; resolution++) {
    uint16_t color = tft.color565((resolution * 20) % 255, (resolution * 40) % 255, (resolution * 60) % 255);
    tessellateSurface(resolution, color);
  }

  delay(50);  // Control the speed of animation
}
