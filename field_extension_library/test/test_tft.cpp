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
  return deg * FE::pi() / FE(180.0f);
}

FE sinFE(const FE& x) { return sin(x); }
FE cosFE(const FE& x) { return cos(x); }

FE computeSunElevation(const FE& latitudeRad, int year, int month, int day, int hour, int min, int sec) {
  static const int mdays[] = { 0,31,28,31,30,31,30,31,31,30,31,30,31 };
  int N=day;
  for(int m=1; m<month; m++) N += mdays[m];

  FE Nfe{float(N)};
  FE twoPi = FE::pi() * FE(2.0f);
  FE decl = FE(23.44f) * sinFE( twoPi * (Nfe + FE(284.0f)) / FE(365.0f) );
  decl = toRadiansFE(decl);

  float hdec = hour + min/60.0f + sec/3600.0f;
  FE H = toRadiansFE( FE(15.0f) * ( FE(hdec) - FE(12.0f) ) );

  FE sinEl = sinFE(decl)*sinFE(latitudeRad) + cosFE(decl)*cosFE(latitudeRad)*cosFE(H);
  float s = sinEl.toFloat();
  if(s>1) s=1; else if(s<-1) s=-1;
  return FE( asin(s) );
}

int main() {
    std::cout << "Running TFT/GPS Example Test..." << std::endl;
    tft.init();

    float lat = 37.7749;
    float lon = -122.4194;
    int yr = 2023, mo = 10, day = 27;
    int hr = 12, mn = 0, sc = 0;

    FE latFE = toRadiansFE(FE(lat));
    FE elFE = computeSunElevation(latFE, yr, mo, day, hr, mn, sc);
    float elevation = std::asin(elFE.toFloat()) * 180.0 / 3.14159265;

    std::cout << "Lat: " << lat << ", Lon: " << lon << std::endl;
    std::cout << "Sun Elevation: " << elevation << " degrees" << std::endl;

    int x0=20, y0=200, w=280;
    for(int i=0; i<=w; i++){
        float d = (50e3) * i / w;
        float drop = R_earth - std::sqrt(R_earth*R_earth - d*d);
        int yi = y0 - (int)(drop / 200.0);
        tft.drawPixel(x0 + i, yi, 0);
    }

    std::cout << "TFT/GPS Example Test Completed Successfully!" << std::endl;
    return 0;
}
