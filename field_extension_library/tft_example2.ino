#include <TFT_eSPI.h>
#include <TinyGPSPlus.h>
#include "FieldExtension.h"

TFT_eSPI tft = TFT_eSPI(); 
TinyGPSPlus gps;
HardwareSerial GPSserial(2);

// Define screen center
const int screenWidth = 320;
const int screenHeight = 240;
const int centerX = screenWidth / 2;
const int centerY = screenHeight / 2;

using FE = FieldElement<4>;

void setup() {
  Serial.begin(115200);
  GPSserial.begin(9600, SERIAL_8N1, 16, 17); // RX, TX for GPS
  tft.init();
  tft.setRotation(1);
  tft.fillScreen(TFT_BLACK);
  tft.setTextColor(TFT_WHITE);
  tft.setTextSize(1);
}

void loop() {
  while (GPSserial.available()) {
    gps.encode(GPSserial.read());
  }

  if (gps.location.isUpdated()) {
    FE lat(gps.location.lat());
    FE lon(gps.location.lng());

    drawEarthCurve(lat, lon);
    drawSunVector(lat, lon);
  }
}

void drawEarthCurve(FE lat, FE lon) {
  tft.fillScreen(TFT_BLACK);

  FE R = FE(6371.0); // Earth's radius in km (use field element for precision)
  float radius = R.toFloat() * 100.0f / 6371.0f; // scale to screen

  // Projected circle as Earth horizon
  for (int angle = 0; angle < 360; angle += 5) {
    float theta = angle * PI / 180.0;
    float x = centerX + radius * cos(theta);
    float y = centerY + radius * sin(theta) * 0.5; // ellipsoid flattening
    tft.drawPixel((int)x, (int)y, TFT_BLUE);
  }

  tft.setCursor(5, 5);
  tft.print("Lat:");
  tft.print(lat.toFloat(), 6);
  tft.setCursor(5, 15);
  tft.print("Lon:");
  tft.print(lon.toFloat(), 6);
}

void drawSunVector(FE lat, FE lon) {
  // Simple solar declination model using day of year
  int dayOfYear = gps.date.day() + gps.date.month() * 30; // rough estimate
  float declination = 23.44 * cos((360.0 / 365.0) * (dayOfYear - 81) * PI / 180.0);

  FE decl(declination);
  FE hourAngle = FE((gps.time.hour() - 12) * 15.0); // degrees

  // Solar altitude (simplified):
  FE sinAlt = sin(lat) * sin(decl) + cos(lat) * cos(decl) * cos(hourAngle);
  float altitude = asin(sinAlt.toFloat()) * 180.0 / PI;

  // Visualize sun position
  int sunY = centerY - (int)(altitude * 2.0f); // exaggerate for display
  tft.fillCircle(centerX, sunY, 4, TFT_YELLOW);

  tft.setCursor(centerX + 10, sunY - 5);
  tft.setTextColor(TFT_YELLOW);
  tft.print("Sun");
}
