#include <Arduino.h>
#include "KahanEMA.hpp"

// Define the smoothing factor (adjust as needed)
const float ALPHA = 0.1;

// Create a KahanEMA object
KahanEMA sensorEMA(ALPHA);

void setup() {
    Serial.begin(115200);
    delay(1000); // Give serial monitor time to connect
    Serial.println("ESP32 Kahan Exponential Averaging Example");
}

void loop() {
    // Simulate reading a sensor value
    // Replace this with your actual sensor reading code
    float sensorReading = analogRead(GPIO_NUM_34) + random(-10, 10) * 0.1; // Example with noise on GPIO 34

    // Update the EMA with the new reading
    float filteredValue = sensorEMA.update(sensorReading);

    // Print the raw and filtered values
    Serial.print("Raw: ");
    Serial.print(sensorReading);
    Serial.print(", Filtered: ");
    Serial.println(filteredValue);

    delay(100); // Adjust the delay based on your sampling rate
}
