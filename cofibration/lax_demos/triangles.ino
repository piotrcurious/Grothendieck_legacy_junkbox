#include <TFT_eSPI.h>  // Include TFT_eSPI library

TFT_eSPI tft = TFT_eSPI(); // Create TFT instance

// Screen dimensions
#define SCREEN_WIDTH  240
#define SCREEN_HEIGHT 320

// Parameters for tessellation
float a = 1.0, b = 1.0;  // Coefficients for elliptic curves
float u_start = -1.0, u_end = 1.0, v_start = -1.0, v_end = 1.0;  // Parameter space
int num_u = 6, num_v = 6;  // Tessellation resolution

// Function to compute points on the elliptic surface
void computeEllipticSurface(float u, float v, float &x, float &y) {
  x = a * u * sqrt(1 - v * v);  // X-coordinate
  y = b * v * sqrt(1 - u * u);  // Y-coordinate
}

// Draw a triangle using tessellated points
void drawTriangle(float x1, float y1, float x2, float y2, float x3, float y3, uint16_t color) {
  tft.drawLine(x1, y1, x2, y2, color);
  tft.drawLine(x2, y2, x3, y3, color);
  tft.drawLine(x3, y3, x1, y1, color);
}

// Tessellation using parameter space
void tessellateSurface() {
  float du = (u_end - u_start) / num_u;
  float dv = (v_end - v_start) / num_v;

  for (int i = 0; i < num_u; i++) {
    for (int j = 0; j < num_v; j++) {
      float u1 = u_start + i * du, v1 = v_start + j * dv;
      float u2 = u_start + (i + 1) * du, v2 = v_start + j * dv;
      float u3 = u_start + i * du, v3 = v_start + (j + 1) * dv;

      float x1, y1, x2, y2, x3, y3;
      computeEllipticSurface(u1, v1, x1, y1);
      computeEllipticSurface(u2, v2, x2, y2);
      computeEllipticSurface(u3, v3, x3, y3);

      // Map coordinates to screen
      x1 = map(x1, -a, a, 0, SCREEN_WIDTH);
      y1 = map(y1, -b, b, 0, SCREEN_HEIGHT);
      x2 = map(x2, -a, a, 0, SCREEN_WIDTH);
      y2 = map(y2, -b, b, 0, SCREEN_HEIGHT);
      x3 = map(x3, -a, a, 0, SCREEN_WIDTH);
      y3 = map(y3, -b, b, 0, SCREEN_HEIGHT);

      // Draw triangle
      drawTriangle(x1, y1, x2, y2, x3, y3, TFT_CYAN);
    }
  }
}

void setup() {
  tft.init();
  tft.setRotation(1);
  tft.fillScreen(TFT_BLACK);

  // Display tessellation
  tessellateSurface();
}

void loop() {
  // Nothing to do in the loop
}
