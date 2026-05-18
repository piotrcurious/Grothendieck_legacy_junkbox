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
  FE R = FE(6371.0f);
  float radius = R.toFloat() * 100.0f / 6371.0f;

  for (int angle = 0; angle < 360; angle += 45) {
    FE theta = (float)angle * FE::pi() / 180.0f;
    float x = centerX + radius * cos(theta).toFloat();
    float y = centerY + radius * sin(theta).toFloat() * 0.5f;
    tft.drawPixel((int)x, (int)y, 0);
  }
}

void drawSunVector(FE lat, FE lon) {
  int dayOfYear = 300;
  float declination = 23.44f * std::cos((360.0f / 365.0f) * (dayOfYear - 81) * M_PI / 180.0f);

  FE decl(declination);
  FE hourAngle = 15.0f * (12.0f - 12.0f);

  FE sinAlt = sin(lat) * sin(decl) + cos(lat) * cos(decl) * cos(hourAngle);
  FE alt = asin(sinAlt);
  float altitude = (alt * 180.0f / FE::pi()).toFloat();
  std::cout << "Sun Altitude (Refined Example 2): " << altitude << " degrees" << std::endl;

  int sunY = centerY - (int)(altitude * 2.0f);
  tft.fillCircle(centerX, sunY, 4, 0);
}

int main() {
    std::cout << "Running Refined TFT/GPS Example 2 Test..." << std::endl;
    tft.init();

    FE lat(37.7749f);
    FE lon(-122.4194f);

    drawEarthCurve(lat, lon);
    drawSunVector(lat, lon);

    std::cout << "Refined TFT/GPS Example 2 Test Completed Successfully!" << std::endl;
    return 0;
}
