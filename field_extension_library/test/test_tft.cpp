#include "Arduino.h"
#include "TFT_eSPI.h"
#include "SPI.h"
#include "TinyGPSPlus.h"
#include "../FieldExtension.h"
#include <iostream>

using FE = FieldElement4;

TFT_eSPI tft = TFT_eSPI(320, 240);
TinyGPSPlus gps;
HardwareSerial SerialGPS(1);

const float R_earth = 6371000.0;

FE toRadiansFE(const FE& deg) {
  return deg * FE::pi() / 180.0f;
}

FE computeSunElevation(const FE& latitudeRad, int year, int month, int day, int hour, int min, int sec) {
  static const int mdays[] = { 0,31,28,31,30,31,30,31,31,30,31,30,31 };
  int N=day;
  for(int m=1; m<month; m++) N += mdays[m];

  FE Nfe{float(N)};
  FE twoPi = FE::pi() * 2.0f;
  FE decl = 23.44f * sin( twoPi * (Nfe + 284.0f) / 365.0f );
  decl = toRadiansFE(decl);

  float hdec = hour + min/60.0f + sec/3600.0f;
  FE H = toRadiansFE( 15.0f * ( hdec - 12.0f ) );

  FE sinEl = sin(decl)*sin(latitudeRad) + cos(decl)*cos(latitudeRad)*cos(H);
  return asin(sinEl);
}

int main() {
    std::cout << "Running Refined TFT/GPS Example Test..." << std::endl;
    tft.init();

    float lat = 37.7749;
    float lon = -122.4194;
    int yr = 2023, mo = 10, day = 27;
    int hr = 12, mn = 0, sc = 0;

    FE latFE = toRadiansFE(FE(lat));
    FE elFE = computeSunElevation(latFE, yr, mo, day, hr, mn, sc);
    float elevation = (elFE * 180.0f / FE::pi()).toFloat();

    std::cout << "Lat: " << lat << ", Lon: " << lon << std::endl;
    std::cout << "Sun Elevation: " << elevation << " degrees" << std::endl;

    int x0=20, y0=200, w=280;
    for(int i=0; i<=w; i++){
        float d = (50e3) * i / w;
        float drop = R_earth - std::sqrt(R_earth*R_earth - d*d);
        int yi = y0 - (int)(drop / 200.0);
        tft.drawPixel(x0 + i, yi, 0);
    }

    std::cout << "Refined TFT/GPS Example Test Completed Successfully!" << std::endl;
    return 0;
}
