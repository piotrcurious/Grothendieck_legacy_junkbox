#include <TFT_eSPI.h>
#include <math.h>

TFT_eSPI tft = TFT_eSPI();

#define SCREEN_WIDTH  240
#define SCREEN_HEIGHT 320

// Animation parameters
float time = 0.0;
int maxResolution = 12;  // Maximum tessellation resolution
int minResolution = 4;   // Minimum tessellation resolution
float perturbationAmplitude = 0.2;  // Amplitude of perturbations

// Function to compute a perturbed surface (sinusoidal surface)
void computePerturbedSurface(float u, float v, float &x, float &y) {
  // Define a sine-wave surface perturbed with time-based perturbations
  x = u;
  y = sin(u + time) + perturbationAmplitude * sin(3 * v + time);  // Perturbation in v direction
  
  // Add additional complex perturbations for more variation
  x += perturbationAmplitude * cos(4 * u + time * 0.5);
  y += perturbationAmplitude * cos(2 * v - time * 0.5);
}

// Map coordinates to screen
void mapToScreen(float x, float y, float &screenX, float &screenY) {
  screenX = map(x, -2.0, 2.0, 20, SCREEN_WIDTH - 20);
  screenY = map(y, -2.0, 2.0, 20, SCREEN_HEIGHT - 20);
}

// Draw a triangle
void drawTriangle(float x1, float y1, float x2, float y2, float x3, float y3, uint16_t color) {
  tft.drawLine(x1, y1, x2, y2, color);
  tft.drawLine(x2, y2, x3, y3, color);
  tft.drawLine(x3, y3, x1, y1, color);
}

// Tessellate the surface with triangles
void tessellateSurface(int resolution, uint16_t color) {
  float du = 2 * M_PI / resolution;  // Step size in u-direction
  float dv = 2 * M_PI / resolution;  // Step size in v-direction

  for (int i = 0; i < resolution; i++) {
    for (int j = 0; j < resolution; j++) {
      // Compute (u, v) values for the current tessellation grid
      float u1 = i * du, v1 = j * dv;
      float u2 = (i + 1) * du, v2 = v1;
      float u3 = u1, v3 = (j + 1) * dv;
      float u4 = u2, v4 = v3;

      // Compute surface points
      float x1, y1, x2, y2, x3, y3, x4, y4;
      computePerturbedSurface(u1, v1, x1, y1);
      computePerturbedSurface(u2, v2, x2, y2);
      computePerturbedSurface(u3, v3, x3, y3);
      computePerturbedSurface(u4, v4, x4, y4);

      // Map the surface points to screen space
      mapToScreen(x1, y1, x1, y1);
      mapToScreen(x2, y2, x2, y2);
      mapToScreen(x3, y3, x3, y3);
      mapToScreen(x4, y4, x4, y4);

      // Draw tessellation (triangles or quadrilaterals)
      drawTriangle(x1, y1, x2, y2, x3, y3, color);
      drawTriangle(x3, y3, x2, y2, x4, y4, color);
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

  // Tessellate with varying resolution
  for (int resolution = minResolution; resolution <= maxResolution; resolution++) {
    uint16_t color = tft.color565((resolution * 20) % 255, (resolution * 40) % 255, (resolution * 60) % 255);
    tessellateSurface(resolution, color);
  }

  delay(50); // Control animation speed
}
