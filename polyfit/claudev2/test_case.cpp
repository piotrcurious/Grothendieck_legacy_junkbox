
#include "polyfit/claudev2/src/arduino_polyfit.hpp"
#include "polyfit/claudev2/src/mock_arduino.hpp"

using namespace polyfit;

void setup() {
    float val = 0.5f;
    GaloisActionExtractor extractor(2);
    float frob_features[2];
    float cyc_features[4];

    extractor.extract_frobenius_orbit(val, frob_features);
    extractor.extract_cyclotomic(val, cyc_features);

    Serial.print("Frob: ");
    for(int i=0; i<2; ++i) { Serial.print(frob_features[i]); Serial.print(" "); }
    Serial.println("");

    Serial.print("Cyc: ");
    for(int i=0; i<4; ++i) { Serial.print(cyc_features[i]); Serial.print(" "); }
    Serial.println("");
}

void loop() {}
