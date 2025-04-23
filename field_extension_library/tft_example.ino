#include <TFT_eSPI.h>          // TFT library
#include <SPI.h>
#include <TinyGPSPlus.h>       // GPS parser
#include "FieldExtension.h"    // your FieldElement<N> library

// — types for convenience —
using FE = FieldElement4;      // basis [1, π, e, √2]

// TFT and GPS objects
TFT_eSPI tft = TFT_eSPI(320, 240);
TinyGPSPlus gps;
HardwareSerial SerialGPS(1);    // use UART1 for GPS

// constants
const float R_earth = 6371000.0;    // meters

// forward declarations
FE toRadiansFE(const FE& deg);
FE sinFE(const FE& x) { return sin(x); }
FE cosFE(const FE& x) { return cos(x); }

/**
 * Compute sun elevation angle (in radians) using a simplified solar geometry:
 *   sin(el) = sin(δ)·sin(φ) + cos(δ)·cos(φ)·cos(H)
 * where
 *   δ = solar declination ≈ 23.44°·sin(2π (N+284)/365)
 *   H = hour angle = 15°·(hour_decimal−12)
 */
FE computeSunElevation(const FE& latitudeRad, int year, int month, int day, int hour, int min, int sec) {
  // 1) Day of year N
  int K = TinyGPSPlus::date.value().day;           // fallback
  // For brevity, approximate N from month/day (not accounting leap) 
  static const int mdays[] = { 0,31,28,31,30,31,30,31,31,30,31,30,31 };
  int N=day;
  for(int m=1; m<month; m++) N += mdays[m];

  // 2) solar declination δ
  FE Nfe(float(N));
  FE twoPi = FE::pi() * FE(2.0f);
  FE decl = FE(23.44f) * sinFE( twoPi * (Nfe + FE(284.0f)) / FE(365.0f) );
  decl = toRadiansFE(decl);

  // 3) hour angle H
  float hdec = hour + min/60.0f + sec/3600.0f;
  FE H = toRadiansFE( FE(15.0f) * ( FE(hdec) - FE(12.0f) ) );

  // 4) compute sin(el)
  FE sinEl = sinFE(decl)*sinFE(latitudeRad) + cosFE(decl)*cosFE(latitudeRad)*cosFE(H);
  // clamp
  float s = sinEl.toFloat();
  if(s>1) s=1; else if(s<-1) s=-1;
  // elevation
  return FE( asinh(s) ); // or just return arcsin: FE(asin(s))
}

FE toRadiansFE(const FE& deg) {
  return deg * FE::pi() / FE(180.0f);
}

void setup() {
  Serial.begin(115200);
  SerialGPS.begin(9600, SERIAL_8N1, 16, 17); // rx=16, tx=17

  tft.init();
  tft.setRotation(1);
  tft.fillScreen(TFT_BLACK);
  tft.setTextSize(1);
  tft.setTextColor(TFT_WHITE);
}

void loop() {
  // feed GPS parser
  while (SerialGPS.available()) {
    gps.encode(SerialGPS.read());
  }
  if (!gps.location.isValid() || !gps.date.isValid() || !gps.time.isValid()) {
    tft.setCursor(0, 0);
    tft.print("Waiting for GPS fix...");
    delay(500);
    return;
  }

  // clear display
  tft.fillScreen(TFT_BLACK);

  // read lat/lon/time
  float lat = gps.location.lat();
  float lon = gps.location.lng();
  int yr = gps.date.year(), mo = gps.date.month(), day = gps.date.day();
  int hr = gps.time.hour(), mn = gps.time.minute(), sc = gps.time.second();

  // convert to FE
  FE latFE = toRadiansFE(FE(lat));
  FE lonFE = toRadiansFE(FE(lon));

  // compute sun elevation
  FE elFE = computeSunElevation(latFE, yr, mo, day, hr, mn, sc);
  float elevation = asinh(elFE.toFloat()) * 180.0 / 3.14159265; 
      // if using arcsin: elevation = asin(elFE.toFloat())*180/PI;

  // draw earth‐curvature graph
  // x axis: distance 0…50 km, y axis: drop due to curvature
  int x0=20, y0=200, w=280, h=150;
  tft.drawRect(x0,y0-h, w, h, TFT_WHITE);
  tft.setCursor(x0, y0-h-10);
  tft.printf("Curvature drop vs distance");

  for(int i=0; i<=w; i++){
    float d = (50e3) * i / w;                  // meters
    float drop = R_earth - sqrt(R_earth*R_earth - d*d);
    int yi = y0 - int(drop / (200.0) );        // vertical scale: 200 m per px
    tft.drawPixel(x0 + i, yi, TFT_CYAN);
  }

  // plot sun position on top: map elevation to y
  int sunX = x0 + w/2;                        // center for illustration
  int sunY = map(constrain(elevation, -90,90), -90, 90, y0, y0-h);
  tft.fillCircle(sunX, sunY, 5, TFT_YELLOW);
  tft.setCursor(10, 10);
  tft.printf("Lat:%.5f\nLon:%.5f\nSun El:%.1f°", lat, lon, elevation);

  delay(2000);
}
