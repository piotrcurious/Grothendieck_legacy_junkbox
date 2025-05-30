#include <TFT_eSPI.h>  // Include TFT_eSPI library

TFT_eSPI tft = TFT_eSPI(); // Create TFT instance

// Screen dimensions
#define SCREEN_WIDTH  240
#define SCREEN_HEIGHT 320

// Parameters for tessellation
float a = 1.0, b = 1.0;  // Coefficients for elliptic curves
float u_start = -1.0, u_end = 1.0, v_start = -1.0, v_end = 1.0;  // Parameter space
int num_u = 8, num_v = 8;  // Tessellation resolution

// Function to compute points on the elliptic surface
void computeEllipticSurface(float u, float v, float &x, float &y) {
  x = a * u * sqrt(1 - v * v);  // X-coordinate
  y = b * v * sqrt(1 - u * u);  // Y-coordinate
}

// Map algebraic coordinates to screen
void mapToScreen(float &x, float &y) {
  x = map(x, -a, a, 0, SCREEN_WIDTH);
  y = map(y, -b, b, 0, SCREEN_HEIGHT);
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

// Tessellate the surface with multiple shapes
void tessellateSurface() {
  float du = (u_end - u_start) / num_u;
  float dv = (v_end - v_start) / num_v;

  for (int i = 0; i < num_u; i++) {
    for (int j = 0; j < num_v; j++) {
      // Parameters for the current grid cell
      float u1 = u_start + i * du, v1 = v_start + j * dv;
      float u2 = u_start + (i + 1) * du, v2 = v_start + j * dv;
      float u3 = u_start + i * du, v3 = v_start + (j + 1) * dv;
      float u4 = u_start + (i + 1) * du, v4 = v_start + (j + 1) * dv;

      // Compute screen coordinates
      float x1, y1, x2, y2, x3, y3, x4, y4;
      computeEllipticSurface(u1, v1, x1, y1);
      computeEllipticSurface(u2, v2, x2, y2);
      computeEllipticSurface(u3, v3, x3, y3);
      computeEllipticSurface(u4, v4, x4, y4);
      mapToScreen(x1, y1);
      mapToScreen(x2, y2);
      mapToScreen(x3, y3);
      mapToScreen(x4, y4);

      // Draw tessellated shapes
      if ((i + j) % 3 == 0) {
        drawTriangle(x1, y1, x2, y2, x3, y3, TFT_RED);
      } else if ((i + j) % 3 == 1) {
        drawRectangle(x1, y1, x2, y2, x4, y4, x3, y3, TFT_GREEN);
      } else {
        float cx = (x1 + x2 + x3 + x4) / 4;
        float cy = (y1 + y2 + y3 + y4) / 4;
        float radius = min(abs(x2 - x1), abs(y3 - y1)) / 2;
        drawHexagon(cx, cy, radius, TFT_BLUE);
      }
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
