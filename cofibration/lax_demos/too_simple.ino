#include <TFT_eSPI.h>
#include <SPI.h>

TFT_eSPI tft = TFT_eSPI();

// Constants for display
const int SCREEN_WIDTH = 320;
const int SCREEN_HEIGHT = 240;
const int CENTER_X = SCREEN_WIDTH / 2;
const int CENTER_Y = SCREEN_HEIGHT / 2;

// Parameters for visualization
float t = 0.0;  // Time parameter
const float dt = 0.05;  // Time step
const int NUM_POINTS = 50;

// Colors
const uint16_t BACKGROUND = TFT_BLACK;
const uint16_t MAPPING_COLOR = TFT_CYAN;
const uint16_t COFIB_COLOR = TFT_MAGENTA;
const uint16_t SPACE_COLOR = TFT_GREEN;

// Structure to represent a point in projective space
struct ProjectivePoint {
  float x, y, z;
};

void setup() {
  tft.init();
  tft.setRotation(1);  // Landscape
  tft.fillScreen(BACKGROUND);
  tft.setTextColor(TFT_WHITE, BACKGROUND);
}

// Map a point from projective space to screen coordinates
void mapToScreen(float px, float py, float pz, int16_t& sx, int16_t& sy) {
  // Perspective projection
  float scale = 100.0 / (pz + 3.0);
  sx = CENTER_X + (int16_t)(px * scale);
  sy = CENTER_Y + (int16_t)(py * scale);
}

// Generate points on an algebraic curve (elliptic curve example)
ProjectivePoint getCurvePoint(float t) {
  // Parametric form of an elliptic curve: y^2 = x^3 - x
  float x = 2 * cos(t);
  float y = 2 * sin(t);
  return {x, y, 1.0};
}

// Generate points in the cofibration space
ProjectivePoint getCofibrationPoint(float t, float s) {
  // Mapping cylinder construction
  float x = 2 * cos(t) * (1 - s);
  float y = 2 * sin(t) * (1 - s);
  float z = 1.0 + s;
  return {x, y, z};
}

void drawCofibrationSequence() {
  tft.fillScreen(BACKGROUND);
  
  // Draw title
  tft.setTextSize(1);
  tft.drawString("Algebraic Geometry & Cofibration", 10, 10);
  
  // Draw base space (elliptic curve)
  for (int i = 0; i < NUM_POINTS; i++) {
    float param = (float)i / NUM_POINTS * 2 * PI;
    ProjectivePoint p = getCurvePoint(param);
    int16_t sx, sy;
    mapToScreen(p.x, p.y, p.z, sx, sy);
    
    if (i > 0) {
      ProjectivePoint prev = getCurvePoint((float)(i-1) / NUM_POINTS * 2 * PI);
      int16_t prev_sx, prev_sy;
      mapToScreen(prev.x, prev.y, prev.z, prev_sx, prev_sy);
      tft.drawLine(prev_sx, prev_sy, sx, sy, SPACE_COLOR);
    }
  }
  
  // Draw cofibration space
  for (int i = 0; i < NUM_POINTS; i++) {
    float theta = (float)i / NUM_POINTS * 2 * PI;
    for (int j = 0; j < 5; j++) {
      float s = (float)j / 4.0;
      ProjectivePoint p = getCofibrationPoint(theta, s);
      int16_t sx, sy;
      mapToScreen(p.x, p.y, p.z, sx, sy);
      
      if (j > 0) {
        ProjectivePoint prev = getCofibrationPoint(theta, (float)(j-1) / 4.0);
        int16_t prev_sx, prev_sy;
        mapToScreen(prev.x, prev.y, prev.z, prev_sx, prev_sy);
        tft.drawLine(prev_sx, prev_sy, sx, sy, COFIB_COLOR);
      }
    }
  }
  
  // Draw mapping between spaces
  for (int i = 0; i < NUM_POINTS; i++) {
    float param = (float)i / NUM_POINTS * 2 * PI;
    ProjectivePoint base = getCurvePoint(param);
    ProjectivePoint cofib = getCofibrationPoint(param, sin(t) * 0.5 + 0.5);
    
    int16_t base_sx, base_sy, cofib_sx, cofib_sy;
    mapToScreen(base.x, base.y, base.z, base_sx, base_sy);
    mapToScreen(cofib.x, cofib.y, cofib.z, cofib_sx, cofib_sy);
    
    tft.drawLine(base_sx, base_sy, cofib_sx, cofib_sy, MAPPING_COLOR);
  }
}

void loop() {
  drawCofibrationSequence();
  t += dt;
  if (t > 2 * PI) t = 0;
  delay(50);
}
