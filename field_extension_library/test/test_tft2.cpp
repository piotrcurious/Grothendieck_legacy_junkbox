#include "Arduino.h"
#include "TFT_eSPI.h"
#include "TinyGPSPlus.h"
#include "../FieldExtension.h"
#include <iostream>

using FE = FieldElement<4>;

TFT_eSPI tft = TFT_eSPI();
TinyGPSPlus gps;

const int screenWidth = 320;
const int screenHeight = 240;
const int centerX = screenWidth / 2;
const int centerY = screenHeight / 2;

void drawEarthCurve(FE lat, FE lon) {
  FE R = FE(6371.0);
  float radius = R.toFloat() * 100.0f / 6371.0f;

  for (int angle = 0; angle < 360; angle += 45) {
    float theta = angle * PI / 180.0;
    float x = centerX + radius * std::cos(theta);
    float y = centerY + radius * std::sin(theta) * 0.5;
    tft.drawPixel((int)x, (int)y, TFT_BLUE);
  }
}

void drawSunVector(FE lat, FE lon) {
  int dayOfYear = 300;
  float declination = 23.44 * std::cos((360.0 / 365.0) * (dayOfYear - 81) * PI / 180.0);

  FE decl(declination);
  FE hourAngle = FE((12 - 12) * 15.0);

  FE sinAlt = sin(lat) * sin(decl) + cos(lat) * cos(decl) * cos(hourAngle);
  float altitude = std::asin(sinAlt.toFloat()) * 180.0 / PI;
  std::cout << "Sun Altitude (Example 2): " << altitude << " degrees" << std::endl;

  int sunY = centerY - (int)(altitude * 2.0f);
  tft.fillCircle(centerX, sunY, 4, TFT_YELLOW);
}

int main() {
    std::cout << "Running TFT/GPS Example 2 Test..." << std::endl;
    tft.init();

    FE lat(37.7749);
    FE lon(-122.4194);

    drawEarthCurve(lat, lon);
    drawSunVector(lat, lon);

    std::cout << "TFT/GPS Example 2 Test Completed Successfully!" << std::endl;
    return 0;
}
