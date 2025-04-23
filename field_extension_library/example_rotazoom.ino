// RotazoomerDemo.ino
// Demonstrates FieldExtension + TFT_eSPI on ESP32 (320×240)  
#include <Arduino.h>
#include <TFT_eSPI.h>            // TFT library by Bodmer 0
#include "FieldExtension.h"      // Your field-extension header

// TFT instance
TFT_eSPI tft = TFT_eSPI();        // Invoke custom User_Setup.h for 320×240

// Screen dimensions
static const int16_t W = 320;
static const int16_t H = 240;

// Center of screen
static const float CX = W / 2.0f;
static const float CY = H / 2.0f;

// Field-extension type with [1, π, e, √2]
using FE = FieldElement4;

// Pre-define a composite angle element: 1·π + 0.5·e + 0.2·√2
FE baseAngle()
{
  FE a;
  a.setCoefficient(1, 1.0f);   // π
  a.setCoefficient(2, 0.5f);   // e
  a.setCoefficient(3, 0.2f);   // √2
  return a;
}

void setup() {
  tft.init();
  tft.setRotation(1);           // Landscape 1
  tft.fillScreen(TFT_BLACK);
}

void loop() {
  static float t = 0.0f;
  t += 0.02f;                    // time step

  // Build an angle FE = base + t·1
  FE ang = baseAngle();
  ang.setCoefficient(0, t);      // add constant term = t

  // Compute sin & cos in the field extension (Taylor path)
  FE s = sin(ang);
  FE c = cos(ang);

  // Convert to floats for pixel mapping
  float fs = s.toFloat();
  float fc = c.toFloat();

  // Compute zoom factor as 1 + 0.5·sin(t/2)
  float zoom = 1.0f + 0.5f * sin(t * 0.5f);

  // Draw rotazoomer
  for (int16_t y = 0; y < H; y++) {
    for (int16_t x = 0; x < W; x++) {
      // Translate to center
      float u = (x - CX) * zoom;
      float v = (y - CY) * zoom;
      // Rotate: x' = u·c – v·s, y' = u·s + v·c
      float xr = u * fc - v * fs;
      float yr = u * fs + v * fc;
      // Simple color from cosine waves
      uint8_t r = (uint8_t)(128 + 127 * cos(xr * 0.05f));
      uint8_t g = (uint8_t)(128 + 127 * cos(yr * 0.05f));
      uint8_t b = (uint8_t)(128 + 127 * cos((xr+yr) * 0.025f));
      uint16_t color = tft.color565(r, g, b);
      tft.drawPixel(x, y, color);
    }
  }

  // Loop animation
}
